#!/usr/bin/env python3
"""
Definitive targeted-lesion analysis for:
"Preserving Information under Structural Change in the Human Connectome"

Primary question
----------------
For equal-magnitude structural perturbations at different cortical locations:
  1) how strongly is baseline predictive information disturbed? (vulnerability)
  2) what fraction can bounded dynamical adaptation recover? (compensability)
  3) how much disturbance remains after adaptation? (residual vulnerability)

The common lesion magnitude is fixed from the preceding calibration:
    ||Delta W||_F = 0.005568...
By default the exact minimum feasible full-incident-lesion norm is recomputed
from W0, so all 68 nodes remain feasible and the value is reproducible.

The adaptive machinery is unchanged from the main manuscript analysis.

Run from repository root:
 python 06b_definitive_targeted_lesions.py \
   --model data/reference/slow_growth_timescale_model_exact.npz \
   --steps 20 \
   --output-dir results/targeted_lesions_definitive

Safe to interrupt with Ctrl+C. Re-running resumes from completed nodes.
"""

import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from model_utils import (
    load_timescale_model, gaussian_cmi_vector, normalized_error,
    theta_jacobian, safe_backtrack, drift, max_real_eig,
    B_STEP, JAC_REFRESH,
)

def full_incident_lesion(W0, k):
    dW = np.zeros_like(W0, dtype=float)
    dW[k, :] = -W0[k, :]
    dW[:, k] = -W0[:, k]
    dW[k, k] = 0.0
    return dW

def matched_lesion(W0, k, target_norm):
    full = full_incident_lesion(W0, k)
    max_norm = np.linalg.norm(full, "fro")
    if max_norm + 1e-15 < target_norm:
        raise ValueError(f"Node {k}: target lesion norm is not feasible.")
    frac = target_norm / max_norm
    return full * frac, frac, max_norm

def run_node(model, node, dW, N):
    W0=model["W0"]; gain=model["gain"]; noise=model["noise_sd"]
    c=model["coupling"]; target=model["target"]; lags=model["lags"]
    per_node=model["per_node"]
    theta0=np.ones(W0.shape[0]); theta=theta0.copy(); J=None
    rows=[]

    for step in range(1, N+1):
        s=step/N
        W=W0+s*dW

        C_no=gaussian_cmi_vector(W,theta0,gain,noise,c,lags)
        err_no=normalized_error(C_no,target)

        C_before=gaussian_cmi_vector(W,theta,gain,noise,c,lags)
        if C_before is None:
            dtheta=np.ones_like(theta)
            dtheta*=B_STEP/np.linalg.norm(dtheta)
            dtheta,C_after=safe_backtrack(W,theta,dtheta,gain,noise,c,lags)
            J=None
        else:
            if J is None or step==1 or (step-1)%JAC_REFRESH==0:
                J=theta_jacobian(W,theta,gain,noise,c,lags)

            if J is None:
                dtheta=np.zeros_like(theta); C_after=C_before
            else:
                sol=lsq_linear(
                    J, -(C_before-target),
                    bounds=(-per_node,per_node),
                    lsmr_tol="auto", max_iter=200
                )
                dtheta=sol.x
                dn=np.linalg.norm(dtheta)
                if dn>B_STEP and dn>0:
                    dtheta*=B_STEP/dn
                dtheta,C_after=safe_backtrack(
                    W,theta,dtheta,gain,noise,c,lags
                )

        theta=theta+dtheta
        err_adapt=normalized_error(C_after,target)
        absolute_recovery=err_no-err_adapt
        K=1-err_adapt/err_no if err_no>1e-12 else np.nan
        margin=-max_real_eig(drift(W,theta,gain,c))

        rows.append(dict(
            node=node, step=step, s=s,
            vulnerability=err_no,
            residual_vulnerability=err_adapt,
            absolute_recovery=absolute_recovery,
            fractional_compensation=K,
            stability_margin=margin,
            dtheta_norm=np.linalg.norm(dtheta),
            cumulative_theta_change=np.linalg.norm(theta-theta0),
        ))
    return pd.DataFrame(rows)

def auc_with_baseline_zero(s, y):
    # Structural progression begins at s=0 with E=0.
    x=np.concatenate(([0.0],np.asarray(s,float)))
    z=np.concatenate(([0.0],np.asarray(y,float)))
    return float(np.trapezoid(z,x))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",
        default="data/reference/slow_growth_timescale_model_exact.npz")
    ap.add_argument("--steps",type=int,default=20)
    ap.add_argument("--output-dir",
        default="results/targeted_lesions_definitive")
    ap.add_argument("--target-norm",type=float,default=None,
        help="Override calibrated common Frobenius lesion norm.")
    ap.add_argument("--nodes",type=int,nargs="*",default=None,
        help="Optional zero-based subset for testing.")
    args=ap.parse_args()

    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    model=load_timescale_model(args.model)
    W0=model["W0"]; labels=model["labels"]; n=W0.shape[0]

    maxnorm=np.array([
        np.linalg.norm(full_incident_lesion(W0,k),"fro") for k in range(n)
    ])
    calibrated=float(np.min(maxnorm))
    target_norm=calibrated if args.target_norm is None else args.target_norm

    if target_norm>calibrated+1e-15 and args.nodes is None:
        infeasible=np.where(maxnorm+1e-15<target_norm)[0]
        raise ValueError(
            f"Target norm {target_norm} is infeasible for nodes {infeasible.tolist()}. "
            f"All-node calibrated maximum is {calibrated}."
        )

    requested=list(range(n)) if args.nodes is None else args.nodes

    summary_path=out/"lesion_summary.csv"
    traj_path=out/"lesion_trajectories.csv"

    if summary_path.exists():
        old_summary=pd.read_csv(summary_path)
        # Guard against accidentally resuming incompatible runs.
        if len(old_summary):
            if "target_norm" in old_summary and not np.allclose(
                old_summary["target_norm"].astype(float),target_norm
            ):
                raise ValueError("Existing output uses a different target norm.")
            if "n_steps" in old_summary and not np.all(
                old_summary["n_steps"].astype(int)==args.steps
            ):
                raise ValueError("Existing output uses a different --steps value.")
        completed=set(old_summary.node.astype(int))
        summaries=old_summary.to_dict("records")
    else:
        completed=set(); summaries=[]

    if traj_path.exists():
        old_traj=pd.read_csv(traj_path)
        old_traj=old_traj[old_traj.node.astype(int).isin(completed)]
        trajectories=[old_traj] if len(old_traj) else []
    else:
        trajectories=[]

    todo=[k for k in requested if k not in completed]
    print(f"Common ||Delta W||F = {target_norm:.10g}")
    print(f"Steps = {args.steps}; requested={len(requested)}; "
          f"complete={len(completed.intersection(requested))}; remaining={len(todo)}")
    if not todo:
        print("All requested nodes already complete."); return

    t0=time.time()
    for count,k in enumerate(todo,1):
        label=labels[k]
        dW,frac,max_feasible=matched_lesion(W0,k,target_norm)
        print(f"[{count:02d}/{len(todo):02d}] {k:02d} {label} | "
              f"incident connectivity removed={100*frac:.2f}%")

        df=run_node(model,k,dW,args.steps)
        df.insert(1,"label",label)
        df["target_norm"]=target_norm
        df["incident_fraction_removed"]=frac
        trajectories.append(df)

        g=df.sort_values("s")
        r=g.iloc[-1]
        auc_no=auc_with_baseline_zero(g.s,g.vulnerability)
        auc_adapt=auc_with_baseline_zero(g.s,g.residual_vulnerability)

        summaries.append(dict(
            node=k,label=label,n_steps=args.steps,target_norm=target_norm,
            incident_fraction_removed=frac,
            max_feasible_lesion_norm=max_feasible,
            node_strength=float(W0[k,:].sum()),
            node_degree=int(np.count_nonzero(W0[k,:])),
            vulnerability=float(r.vulnerability),
            compensability=float(r.fractional_compensation),
            residual_vulnerability=float(r.residual_vulnerability),
            absolute_recovery=float(r.absolute_recovery),
            integrated_vulnerability=auc_no,
            integrated_residual_vulnerability=auc_adapt,
            integrated_absolute_recovery=auc_no-auc_adapt,
            endpoint_stability_margin=float(r.stability_margin),
            cumulative_theta_change=float(r.cumulative_theta_change),
        ))

        pd.concat(trajectories,ignore_index=True).to_csv(traj_path,index=False)
        pd.DataFrame(summaries).to_csv(summary_path,index=False)

        elapsed=(time.time()-t0)/60
        print(f"    V={r.vulnerability:.6g} | "
              f"K={r.fractional_compensation:.3f} | "
              f"R={r.residual_vulnerability:.6g} | "
              f"elapsed={elapsed:.1f} min")

    s=pd.DataFrame(summaries)
    s.to_csv(summary_path,index=False)
    s.sort_values("vulnerability",ascending=False).to_csv(
        out/"ranked_vulnerability.csv",index=False)
    s.sort_values("residual_vulnerability",ascending=False).to_csv(
        out/"ranked_residual_vulnerability.csv",index=False)
    s.sort_values("compensability",ascending=False).to_csv(
        out/"ranked_compensability.csv",index=False)

    print("\nMost vulnerable equal-magnitude lesions:")
    print(s.sort_values("vulnerability",ascending=False)[
        ["node","label","vulnerability","compensability",
         "residual_vulnerability"]].head(10).to_string(index=False))
    print("\nLargest residual disruption after adaptation:")
    print(s.sort_values("residual_vulnerability",ascending=False)[
        ["node","label","vulnerability","compensability",
         "residual_vulnerability"]].head(10).to_string(index=False))
    print("\nDefinitive targeted-lesion analysis complete.")

if __name__=="__main__":
    main()

#!/usr/bin/env python3
"""
Proportional targeted-lesion control.

Control question:
Does regional heterogeneity persist when every cortical node loses the SAME
FRACTION of its incident structural connectivity, rather than the same
absolute Frobenius magnitude?

Default fraction = 0.10 (10%). This is a distinct normalization/control, not
a model of literal 10% tissue loss.

Uses the same OU/CMI/adaptation machinery as the main manuscript and the
definitive equal-magnitude lesion analysis.

Run:
 python 06c_proportional_lesion_control.py \
   --model data/reference/slow_growth_timescale_model_exact.npz \
   --fraction 0.10 --steps 20 \
   --output-dir results/targeted_lesions_proportional

Safe to interrupt; rerunning resumes completed nodes.
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

def proportional_lesion(W0,k,fraction):
    dW=np.zeros_like(W0,dtype=float)
    dW[k,:]=-fraction*W0[k,:]
    dW[:,k]=-fraction*W0[:,k]
    dW[k,k]=0.0
    return dW

def run_node(m,k,dW,N):
    W0=m["W0"]; gain=m["gain"]; noise=m["noise_sd"]; c=m["coupling"]
    target=m["target"]; lags=m["lags"]; per_node=m["per_node"]
    theta0=np.ones(W0.shape[0]); theta=theta0.copy(); J=None; rows=[]
    for step in range(1,N+1):
        s=step/N; W=W0+s*dW
        Cno=gaussian_cmi_vector(W,theta0,gain,noise,c,lags)
        eno=normalized_error(Cno,target)
        Cb=gaussian_cmi_vector(W,theta,gain,noise,c,lags)
        if Cb is None:
            dt=np.ones_like(theta); dt*=B_STEP/np.linalg.norm(dt)
            dt,Ca=safe_backtrack(W,theta,dt,gain,noise,c,lags); J=None
        else:
            if J is None or step==1 or (step-1)%JAC_REFRESH==0:
                J=theta_jacobian(W,theta,gain,noise,c,lags)
            if J is None:
                dt=np.zeros_like(theta); Ca=Cb
            else:
                sol=lsq_linear(J,-(Cb-target),bounds=(-per_node,per_node),
                               lsmr_tol="auto",max_iter=200)
                dt=sol.x; dn=np.linalg.norm(dt)
                if dn>B_STEP and dn>0: dt*=B_STEP/dn
                dt,Ca=safe_backtrack(W,theta,dt,gain,noise,c,lags)
        theta+=dt
        ea=normalized_error(Ca,target)
        K=1-ea/eno if eno>1e-12 else np.nan
        rows.append(dict(node=k,step=step,s=s,vulnerability=eno,
            residual_vulnerability=ea,absolute_recovery=eno-ea,
            fractional_compensation=K,
            stability_margin=-max_real_eig(drift(W,theta,gain,c)),
            dtheta_norm=np.linalg.norm(dt),
            cumulative_theta_change=np.linalg.norm(theta-theta0)))
    return pd.DataFrame(rows)

def auc(s,y):
    return float(np.trapezoid(np.r_[0,np.asarray(y,float)],
                              np.r_[0,np.asarray(s,float)]))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",default="data/reference/slow_growth_timescale_model_exact.npz")
    p.add_argument("--fraction",type=float,default=.10)
    p.add_argument("--steps",type=int,default=20)
    p.add_argument("--output-dir",default="results/targeted_lesions_proportional")
    p.add_argument("--nodes",type=int,nargs="*",default=None)
    a=p.parse_args()
    if not 0<a.fraction<=1: raise ValueError("--fraction must be in (0,1].")
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    m=load_timescale_model(a.model); W0=m["W0"]; labels=m["labels"]; n=W0.shape[0]
    requested=list(range(n)) if a.nodes is None else a.nodes
    sp=out/"lesion_summary.csv"; tp=out/"lesion_trajectories.csv"

    if sp.exists():
        old=pd.read_csv(sp)
        if len(old):
            if not np.allclose(old["incident_fraction_removed"].astype(float),a.fraction):
                raise ValueError("Existing output uses a different lesion fraction.")
            if not np.all(old["n_steps"].astype(int)==a.steps):
                raise ValueError("Existing output uses a different --steps value.")
        done=set(old.node.astype(int)); summaries=old.to_dict("records")
    else: done=set(); summaries=[]
    if tp.exists():
        ot=pd.read_csv(tp); ot=ot[ot.node.astype(int).isin(done)]
        trajectories=[ot] if len(ot) else []
    else: trajectories=[]

    todo=[k for k in requested if k not in done]
    print(f"Proportional lesion: {100*a.fraction:.1f}% incident connectivity; "
          f"steps={a.steps}; remaining={len(todo)}")
    t0=time.time()
    for i,k in enumerate(todo,1):
        dW=proportional_lesion(W0,k,a.fraction)
        norm=float(np.linalg.norm(dW,"fro"))
        print(f"[{i:02d}/{len(todo):02d}] {k:02d} {labels[k]} | ||dW||F={norm:.6g}")
        df=run_node(m,k,dW,a.steps); df.insert(1,"label",labels[k])
        df["incident_fraction_removed"]=a.fraction; df["lesion_norm"]=norm
        trajectories.append(df)
        g=df.sort_values("s"); r=g.iloc[-1]
        an=auc(g.s,g.vulnerability); aa=auc(g.s,g.residual_vulnerability)
        summaries.append(dict(node=k,label=labels[k],n_steps=a.steps,
            incident_fraction_removed=a.fraction,lesion_norm=norm,
            node_strength=float(W0[k,:].sum()),
            node_degree=int(np.count_nonzero(W0[k,:])),
            vulnerability=float(r.vulnerability),
            compensability=float(r.fractional_compensation),
            residual_vulnerability=float(r.residual_vulnerability),
            absolute_recovery=float(r.absolute_recovery),
            integrated_vulnerability=an,
            integrated_residual_vulnerability=aa,
            integrated_absolute_recovery=an-aa,
            endpoint_stability_margin=float(r.stability_margin),
            cumulative_theta_change=float(r.cumulative_theta_change)))
        pd.concat(trajectories,ignore_index=True).to_csv(tp,index=False)
        pd.DataFrame(summaries).to_csv(sp,index=False)
        print(f"    V={r.vulnerability:.6g} | K={r.fractional_compensation:.3f} | "
              f"R={r.residual_vulnerability:.6g} | elapsed={(time.time()-t0)/60:.1f} min")

    s=pd.DataFrame(summaries)
    s.sort_values("vulnerability",ascending=False).to_csv(out/"ranked_vulnerability.csv",index=False)
    s.sort_values("residual_vulnerability",ascending=False).to_csv(out/"ranked_residual_vulnerability.csv",index=False)
    s.sort_values("compensability",ascending=False).to_csv(out/"ranked_compensability.csv",index=False)
    print("\nProportional lesion control analysis complete.")

if __name__=="__main__":
    main()

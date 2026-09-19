#!/usr/bin/env python3
"""Calibrate a common magnitude for targeted cortical lesions.

Tests equal-Frobenius-norm incident-edge lesions across all 68 nodes WITHOUT
adaptation. This chooses a meaningful perturbation magnitude before the
expensive adaptive sweep.

Run:
 python 06a_calibrate_targeted_lesions.py \
   --model data/reference/slow_growth_timescale_model_exact.npz \
   --output-dir results/lesion_calibration
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from model_utils import load_timescale_model, gaussian_cmi_vector, normalized_error, drift, max_real_eig

def full_incident_lesion(W0, k):
    dW=np.zeros_like(W0,dtype=float)
    dW[k,:]=-W0[k,:]; dW[:,k]=-W0[:,k]; dW[k,k]=0
    return dW

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",default="data/reference/slow_growth_timescale_model_exact.npz")
    p.add_argument("--output-dir",default="results/lesion_calibration")
    p.add_argument("--target-norms",type=float,nargs="*",default=None)
    a=p.parse_args()
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    m=load_timescale_model(a.model)
    W0=m["W0"]; labels=m["labels"]; gain=m["gain"]; noise=m["noise_sd"]
    c=m["coupling"]; target=m["target"]; lags=m["lags"]
    n=W0.shape[0]; theta0=np.ones(n)
    full=[full_incident_lesion(W0,k) for k in range(n)]
    maxnorm=np.array([np.linalg.norm(x,"fro") for x in full])

    pd.DataFrame({
      "node":np.arange(n),"label":labels,
      "node_strength":[W0[k,:].sum() for k in range(n)],
      "node_degree":[np.count_nonzero(W0[k,:]) for k in range(n)],
      "max_feasible_lesion_norm":maxnorm
    }).to_csv(out/"lesion_node_feasibility.csv",index=False)

    if a.target_norms:
        candidates=[(np.nan,x) for x in sorted(set(a.target_norms))]
    else:
        qs=[0,.05,.10,.20,.30,.40,.50]
        candidates=[(q,float(np.quantile(maxnorm,q))) for q in qs]

    allrows=[]; summaries=[]
    print("Testing",len(candidates),"common lesion magnitudes across",n,"nodes")
    for q,tn in candidates:
        rows=[]
        for k in range(n):
            feasible=maxnorm[k]+1e-15>=tn and maxnorm[k]>0
            if feasible:
                frac=tn/maxnorm[k]
                W=W0+full[k]*frac
                C=gaussian_cmi_vector(W,theta0,gain,noise,c,lags)
                err=np.nan if C is None else normalized_error(C,target)
                margin=-max_real_eig(drift(W,theta0,gain,c))
            else:
                frac=err=margin=np.nan
            row={"target_norm":tn,"source_quantile":q,"node":k,"label":labels[k],
                 "feasible":feasible,"fraction_incident_weight_removed":frac,
                 "error_nocomp":err,"stability_margin":margin}
            rows.append(row); allrows.append(row)
        d=pd.DataFrame(rows)
        g=d[d.feasible & d.error_nocomp.notna()]
        e=g.error_nocomp.to_numpy()
        s={"target_norm":tn,"source_quantile":q,"n_feasible":int(d.feasible.sum()),
           "fraction_feasible":d.feasible.mean()}
        if len(e):
            s.update(error_min=e.min(),error_q10=np.quantile(e,.1),error_q25=np.quantile(e,.25),
                     error_median=np.median(e),error_q75=np.quantile(e,.75),
                     error_q90=np.quantile(e,.9),error_max=e.max(),
                     fraction_error_gt_1e4=np.mean(e>1e-4),
                     fraction_error_gt_5e4=np.mean(e>5e-4),
                     fraction_error_gt_1e3=np.mean(e>1e-3),
                     minimum_stability_margin=g.stability_margin.min())
        summaries.append(s)
        pd.DataFrame(allrows).to_csv(out/"lesion_calibration_all.csv",index=False)
        pd.DataFrame(summaries).to_csv(out/"lesion_calibration_by_magnitude.csv",index=False)
        print(f"||dW||F={tn:.6g}: feasible={s['n_feasible']}/{n}, "
              f"median E={s.get('error_median',np.nan):.6g}, "
              f"q10={s.get('error_q10',np.nan):.6g}, q90={s.get('error_q90',np.nan):.6g}")
    print(f"\nCalibration complete. Results written to: {out}")

if __name__=="__main__":
    main()

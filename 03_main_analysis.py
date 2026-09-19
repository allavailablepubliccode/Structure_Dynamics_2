#!/usr/bin/env python3
"""Primary rate analysis reported in Table 1 (N=5,10,20,40; full DeltaW endpoint)."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from model_utils import (load_timescale_model, gaussian_cmi_vector, normalized_error,
                         theta_jacobian, safe_backtrack, drift, max_real_eig,
                         B_STEP, JAC_REFRESH)
from scipy.optimize import lsq_linear

N_LIST = [5,10,20,40]


def run_condition(model, N):
    W0=model['W0']; dW=model['DeltaW']; gain=model['gain']; noise=model['noise_sd'];
    c=model['coupling']; target=model['target']; lags=model['lags']; per_node=model['per_node']
    theta0=np.ones(W0.shape[0]); theta=theta0.copy(); J=None; rows=[]
    for step in range(1,N+1):
        s=step/N; W=W0+s*dW
        C_no=gaussian_cmi_vector(W,theta0,gain,noise,c,lags); err_no=normalized_error(C_no,target)
        C_before=gaussian_cmi_vector(W,theta,gain,noise,c,lags)
        if C_before is None:
            dtheta=np.ones_like(theta); dtheta*=B_STEP/np.linalg.norm(dtheta)
            dtheta,C_after=safe_backtrack(W,theta,dtheta,gain,noise,c,lags); J=None
        else:
            if J is None or step==1 or (step-1)%JAC_REFRESH==0:
                J=theta_jacobian(W,theta,gain,noise,c,lags)
            if J is None:
                dtheta=np.zeros_like(theta); C_after=C_before
            else:
                sol=lsq_linear(J,-(C_before-target),bounds=(-per_node,per_node),lsmr_tol='auto',max_iter=200)
                dtheta=sol.x
                dn=np.linalg.norm(dtheta)
                if dn>B_STEP and dn>0: dtheta*=B_STEP/dn
                dtheta,C_after=safe_backtrack(W,theta,dtheta,gain,noise,c,lags)
        theta=theta+dtheta
        err=normalized_error(C_after,target); K=1-err/err_no if err_no>1e-15 else np.nan
        margin=-max_real_eig(drift(W,theta,gain,c))
        rows.append(dict(n_steps=N,step=step,s=s,error_nocomp=err_no,error_comp=err,
                         K=K,stability_margin=margin,dtheta_norm=np.linalg.norm(dtheta),
                         theta_norm=np.linalg.norm(theta-theta0)))
        print(f"N={N:2d} step={step:2d}/{N}: E={err:.6f}, K={K:.6f}, m={margin:.6f}")
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',default='data/generated/slow_growth_timescale_model_exact.npz'); ap.add_argument('--output-dir',default='results/generated/main')
    args=ap.parse_args(); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    model=load_timescale_model(args.model)
    all_df=pd.concat([run_condition(model,N) for N in N_LIST],ignore_index=True)
    all_df.to_csv(out/'main_trajectories.csv',index=False)
    summaries=[]
    for N,g in all_df.groupby('n_steps'):
        g=g.sort_values('s'); x=g.s.to_numpy(); y=g.error_comp.to_numpy()
        integrated=float(np.trapezoid(y,x))
        r=g.iloc[-1]
        summaries.append(dict(n_steps=N,endpoint_error=r.error_comp,fractional_compensation=r.K,
                              integrated_error=integrated,stability_margin=r.stability_margin))
    pd.DataFrame(summaries).to_csv(out/'main_summary.csv',index=False)

if __name__=='__main__': main()

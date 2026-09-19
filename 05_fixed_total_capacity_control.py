#!/usr/bin/env python3
"""Fixed-total-adaptive-capacity control reported in Table 1 (N=5 and N=20)."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from model_utils import (load_timescale_model, gaussian_cmi_vector, theta_jacobian,
                         safe_backtrack, max_real_eig, drift, JAC_REFRESH)

TOTAL_L2=0.12
TOTAL_PER_NODE=0.08
N_LIST=[5,20]


def run(model,N):
    B=TOTAL_L2/N; per_node=TOTAL_PER_NODE/N
    W0=model['W0']; dW=model['DeltaW']; gain=model['gain']; noise=model['noise_sd']; c=model['coupling']; target=model['target']; lags=model['lags']
    theta0=np.ones(W0.shape[0]); theta=theta0.copy(); J=None; rows=[]
    for step in range(1,N+1):
        s=step/N; W=W0+s*dW
        Cno=gaussian_cmi_vector(W,theta0,gain,noise,c,lags); C=gaussian_cmi_vector(W,theta,gain,noise,c,lags)
        eno=np.linalg.norm(Cno-target)/np.linalg.norm(target)
        if J is None or step==1 or (step-1)%JAC_REFRESH==0: J=theta_jacobian(W,theta,gain,noise,c,lags)
        if J is not None:
            sol=lsq_linear(J,-(C-target),bounds=(-per_node,per_node),lsmr_tol='auto',max_iter=200); dtheta=sol.x
            dn=np.linalg.norm(dtheta)
            if dn>B and dn>0: dtheta*=B/dn
            dtheta,_=safe_backtrack(W,theta,dtheta,gain,noise,c,lags); theta+=dtheta
        Cafter=gaussian_cmi_vector(W,theta,gain,noise,c,lags)
        err=np.linalg.norm(Cafter-target)/np.linalg.norm(target); K=1-err/eno
        margin=-max_real_eig(drift(W,theta,gain,c))
        rows.append(dict(n_steps=N,B_step=B,per_node_step=per_node,step=step,s=s,error_nocomp=eno,error_comp=err,K=K,stability_margin=margin))
        print(f"N={N:2d} step={step:2d}/{N}: E={err:.6f}, K={K:.6f}")
    return rows


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',default='data/generated/slow_growth_timescale_model_exact.npz'); ap.add_argument('--output-dir',default='results/generated/fixed_total')
    a=ap.parse_args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    model=load_timescale_model(a.model); rows=[]
    for N in N_LIST: rows.extend(run(model,N))
    df=pd.DataFrame(rows); df.to_csv(out/'fixed_total_trajectories.csv',index=False)
    s=[]
    for N,g in df.groupby('n_steps'):
        r=g.sort_values('step').iloc[-1]
        s.append(dict(n_steps=N,total_L2_budget=TOTAL_L2,total_per_node_budget=TOTAL_PER_NODE,B_step=r.B_step,per_node_step=r.per_node_step,endpoint_error=r.error_comp,fractional_compensation=r.K))
    pd.DataFrame(s).to_csv(out/'fixed_total_summary.csv',index=False)

if __name__=='__main__': main()

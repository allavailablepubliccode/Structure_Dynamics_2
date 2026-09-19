#!/usr/bin/env python3
"""Robustness across the four structural-change patterns reported in Table 1."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from model_utils import (load_timescale_model, gaussian_cmi_vector, theta_jacobian,
                         safe_backtrack, max_real_eig, drift, B_STEP, JAC_REFRESH)

N_LIST=[5,10,20,40]


def norm_to(raw,reference):
    raw=0.5*(raw+raw.T); np.fill_diagonal(raw,0)
    return raw*(np.linalg.norm(reference,'fro')/np.linalg.norm(raw,'fro'))


def build_patterns(W0,primary,struct):
    mask=(W0>0).astype(float); np.fill_diagonal(mask,0)
    D=struct['D'].astype(float); M=struct['M'].astype(float); dmed=float(struct['dmed'])
    return {
        'Primary': primary.copy(),
        'Proportional': norm_to(W0*mask,primary),
        'Distance-dependent': norm_to(W0*np.exp(-D/dmed)*mask,primary),
        'Matching-index-dependent': norm_to(W0*(M+0.05)*mask,primary),
    }


def run(model,dW,name,N):
    W0=model['W0']; gain=model['gain']; noise=model['noise_sd']; c=model['coupling']; target=model['target']; lags=model['lags']; per_node=model['per_node']
    theta0=np.ones(W0.shape[0]); theta=theta0.copy(); J=None; rows=[]
    for step in range(1,N+1):
        s=step/N; W=W0+s*dW
        Cno=gaussian_cmi_vector(W,theta0,gain,noise,c,lags); C=gaussian_cmi_vector(W,theta,gain,noise,c,lags)
        eno=np.linalg.norm(Cno-target)/np.linalg.norm(target)
        if J is None or (step-1)%JAC_REFRESH==0: J=theta_jacobian(W,theta,gain,noise,c,lags)
        if J is not None:
            sol=lsq_linear(J,-(C-target),bounds=(-per_node,per_node),lsmr_tol='auto',max_iter=200); dtheta=sol.x
            dn=np.linalg.norm(dtheta)
            if dn>B_STEP and dn>0: dtheta*=B_STEP/dn
            dtheta,_=safe_backtrack(W,theta,dtheta,gain,noise,c,lags); theta+=dtheta
        Cafter=gaussian_cmi_vector(W,theta,gain,noise,c,lags)
        err=np.linalg.norm(Cafter-target)/np.linalg.norm(target); K=1-err/eno
        margin=-max_real_eig(drift(W,theta,gain,c))
        rows.append(dict(pattern=name,n_steps=N,step=step,s=s,error_comp=err,error_nocomp=eno,K=K,stability_margin=margin))
        print(f"{name:25s} N={N:2d} step={step:2d}/{N}: E={err:.6f}")
    return rows


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',default='data/generated/slow_growth_timescale_model_exact.npz'); ap.add_argument('--structural-model',default='data/generated/slow_growth_structural_model.npz'); ap.add_argument('--output-dir',default='results/generated/patterns')
    a=ap.parse_args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    model=load_timescale_model(a.model); struct=np.load(a.structural_model,allow_pickle=True)
    patterns=build_patterns(model['W0'],model['DeltaW'],struct)
    rows=[]
    for name,dW in patterns.items():
        for N in N_LIST: rows.extend(run(model,dW,name,N))
    df=pd.DataFrame(rows); df.to_csv(out/'pattern_trajectories.csv',index=False)
    summary=[]
    for (name,N),g in df.groupby(['pattern','n_steps']):
        g=g.sort_values('s'); r=g.iloc[-1]
        summary.append(dict(pattern=name,n_steps=N,integrated_error=float(np.trapezoid(g.error_comp,g.s)),endpoint_error=r.error_comp,endpoint_K=r.K,endpoint_margin=r.stability_margin))
    pd.DataFrame(summary).sort_values(['pattern','n_steps']).to_csv(out/'pattern_summary.csv',index=False)

if __name__=='__main__': main()

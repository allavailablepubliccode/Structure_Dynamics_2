#!/usr/bin/env python3
"""Check reconstructed empirical/model files against archived reference files."""
import argparse
from pathlib import Path
import numpy as np

RELEVANT_STRUCT_KEYS=['W0','DeltaW','Q','D','M','coords','labels','C_target','dmed','growth_amp','B_step','per_node','lags']
TIME_KEYS=['W0','DeltaW','labels','gain','noise_sd','global_coupling','C_target','original_growth_amp','original_B_step','original_per_node','lags']
CONS_KEYS=['W','labels','prevalence','mean_fibers_present','gain','noise_sd','global_coupling']

def compare(a,b,keys,tol=1e-10):
    A=np.load(a,allow_pickle=True); B=np.load(b,allow_pickle=True); ok=True
    for k in keys:
        x,y=A[k],B[k]
        if x.dtype.kind in 'USO' or y.dtype.kind in 'USO':
            same=np.array_equal(x,y); print(f'{k:28s} exact={same}'); ok &= same
        else:
            d=float(np.max(np.abs(x-y))) if np.size(x) else 0.0
            print(f'{k:28s} max_abs_diff={d:.3e}'); ok &= d<=tol
    return ok

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--derived-dir',default='data/derived'); ap.add_argument('--generated-dir',default='data/generated'); ap.add_argument('--reference-dir',default='data/reference')
    a=ap.parse_args(); d=Path(a.derived_dir); g=Path(a.generated_dir); r=Path(a.reference_dir)
    print('GROUP CONNECTOME')
    ok1=compare(d/'hcp86_empirical_consensus.npz',r.parent/'derived'/'hcp86_empirical_consensus.npz',CONS_KEYS)
    print('\nSTRUCTURAL MODEL (keys used by manuscript analyses)')
    ok2=compare(g/'slow_growth_structural_model.npz',r/'slow_growth_structural_model.npz',RELEVANT_STRUCT_KEYS)
    print('\nTIMESCALE MODEL')
    ok3=compare(g/'slow_growth_timescale_model_exact.npz',r/'slow_growth_timescale_model_exact.npz',TIME_KEYS)
    print('\nPASS' if ok1 and ok2 and ok3 else '\nFAIL')
    raise SystemExit(0 if ok1 and ok2 and ok3 else 1)
if __name__=='__main__': main()

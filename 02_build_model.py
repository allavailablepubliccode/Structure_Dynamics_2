#!/usr/bin/env python3
"""Construct the structural-change model and exact OU/CMI model used in the manuscript."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from model_utils import gaussian_cmi_vector, matching_index_binary

GROWTH_AMP = 0.20
B_STEP = 0.012
PER_NODE = 0.004
LAGS = np.array([0.5], dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--consensus", default="data/derived/hcp86_empirical_consensus.npz")
    ap.add_argument("--coordinates", default="data/derived/hcp68_group_coordinates.csv")
    ap.add_argument("--output-dir", default="data/generated")
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    c = np.load(args.consensus, allow_pickle=True)
    W0 = c["W"].astype(float); labels = c["labels"].astype(str)
    gain = c["gain"].astype(float); noise = c["noise_sd"].astype(float)
    coupling = float(c["global_coupling"])

    df = pd.read_csv(args.coordinates).set_index("label").loc[labels]
    coords = df[["x","y","z"]].to_numpy(float)
    D = np.linalg.norm(coords[:,None,:]-coords[None,:,:], axis=2)
    M = matching_index_binary(W0)
    mask = W0 > 0; np.fill_diagonal(mask, False)
    dmed = float(np.median(D[mask & (D>0)]))

    factor = (M+0.05)*np.exp(-D/dmed)
    Q = np.zeros_like(W0)
    Q[mask] = factor[mask]/np.mean(factor[mask])  # mean Q over existing directed edges = 1
    DeltaW = GROWTH_AMP * W0 * Q
    DeltaW = 0.5*(DeltaW+DeltaW.T); np.fill_diagonal(DeltaW,0.0)

    theta0 = np.ones(W0.shape[0])
    target = gaussian_cmi_vector(W0, theta0, gain, noise, coupling, tuple(LAGS))
    if target is None:
        raise RuntimeError("Baseline model is unstable")

    np.savez(out/"slow_growth_structural_model.npz", W0=W0, DeltaW=DeltaW, Q=Q, D=D, M=M,
             coords=coords, labels=labels, theta_final=theta0, C_target=target, dmed=dmed,
             growth_amp=GROWTH_AMP, B_step=B_STEP, per_node=PER_NODE, lags=LAGS)
    np.savez(out/"slow_growth_timescale_model_exact.npz", W0=W0, DeltaW=DeltaW, labels=labels,
             gain=gain, noise_sd=noise, global_coupling=coupling, C_target=target,
             original_growth_amp=GROWTH_AMP, original_B_step=B_STEP,
             original_per_node=PER_NODE, lags=LAGS)

    alpha = GROWTH_AMP/np.mean(factor[mask])
    A0 = coupling*np.diag(gain)@W0-np.eye(W0.shape[0])
    print(f"d_med={dmed:.15f}")
    print(f"DeltaW proportionality alpha={alpha:.16f}")
    print(f"baseline leading real eigenvalue={np.max(np.real(np.linalg.eigvals(A0))):.12f}")
    print(f"CMI values={target.size}")

if __name__ == "__main__":
    main()

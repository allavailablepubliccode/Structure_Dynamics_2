#!/usr/bin/env python3
"""
Fast, resumable multi-seed robustness test for PRE-ADAPTATION focal vulnerability.

This tests whether the anatomical pattern in Fig. 1A depends on the arbitrary
regional gain/noise realization. It does NOT run dynamical adaptation.

For each seed:
  - keep W0 fixed;
  - redraw gain_i ~ U(0.85, 1.15);
  - recalibrate coupling so max Re eig(diag(gain) @ W0) = 0.72;
  - redraw noise_sd_i ~ U(0.85, 1.15);
  - recompute baseline CMI;
  - apply the same equal-Frobenius full endpoint lesion to all 68 regions;
  - calculate initial vulnerability.

Each completed seed is saved separately. Re-running the same command skips
completed seeds, so the job can be interrupted with Ctrl-C and resumed.

Run:
  python 07c_fast_seed_robustness.py --n-seeds 100 --workers 6
"""

# Avoid BLAS oversubscription when using several worker processes.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import signal
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from model_utils import gaussian_cmi_vector, normalized_error


def full_incident_lesion(W0, k):
    dW = np.zeros_like(W0, dtype=float)
    dW[k, :] = -W0[k, :]
    dW[:, k] = -W0[:, k]
    dW[k, k] = 0.0
    return dW


def matched_lesions(W0):
    full = [full_incident_lesion(W0, k) for k in range(W0.shape[0])]
    maxnorm = np.asarray([np.linalg.norm(x, "fro") for x in full])
    target_norm = float(np.min(maxnorm))
    lesions = [
        x * (target_norm / maxnorm[k])
        for k, x in enumerate(full)
    ]
    return target_norm, lesions


def hemi(label):
    s = str(label).lower()
    if s.startswith("lh."):
        return "lh"
    if s.startswith("rh."):
        return "rh"
    return ""


def run_seed(seed, W0, labels, lags, lesions, node_strength):
    """Compute endpoint pre-adaptation vulnerability for one random realization."""
    n = W0.shape[0]
    rng = np.random.default_rng(seed)

    # Exact construction used in 01_build_group_connectome.py.
    gain = rng.uniform(0.85, 1.15, n)
    rM = np.max(np.real(np.linalg.eigvals(np.diag(gain) @ W0)))
    coupling = 0.72 / rM
    noise_sd = rng.uniform(0.85, 1.15, n)

    theta0 = np.ones(n)

    target = gaussian_cmi_vector(
        W0, theta0, gain, noise_sd, coupling, tuple(lags)
    )
    if target is None:
        raise RuntimeError(f"Baseline CMI failed for seed {seed}")

    rows = []
    for k in range(n):
        W = W0 + lesions[k]

        C = gaussian_cmi_vector(
            W, theta0, gain, noise_sd, coupling, tuple(lags)
        )
        if C is None:
            vulnerability = np.nan
        else:
            vulnerability = float(normalized_error(C, target))

        rows.append({
            "seed": int(seed),
            "node": int(k),
            "label": str(labels[k]),
            "hemisphere": hemi(labels[k]),
            "node_strength": float(node_strength[k]),
            "vulnerability": vulnerability,
            "coupling": float(coupling),
        })

    df = pd.DataFrame(rows)
    df["vulnerability_rank"] = df["vulnerability"].rank(
        ascending=False, method="average"
    )
    return df


def atomic_write_csv(df, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def valid_checkpoint(path, seed, n_regions):
    try:
        df = pd.read_csv(path)
        return (
            len(df) == n_regions
            and "seed" in df.columns
            and set(df["seed"].astype(int)) == {int(seed)}
            and df["node"].nunique() == n_regions
            and df["vulnerability"].notna().all()
        )
    except Exception:
        return False


def aggregate(completed_dir, expected_seeds, labels, top_n, output_dir):
    frames = []
    for seed in expected_seeds:
        p = completed_dir / f"seed_{seed}.csv"
        if p.exists() and valid_checkpoint(p, seed, len(labels)):
            frames.append(pd.read_csv(p))

    if not frames:
        return None

    results = pd.concat(frames, ignore_index=True)
    completed_seeds = sorted(results["seed"].astype(int).unique())

    # Per-seed summaries.
    seed_rows = []
    for seed, g in results.groupby("seed"):
        rho, _ = spearmanr(g["node_strength"], g["vulnerability"])
        left = g.loc[g["hemisphere"] == "lh", "vulnerability"].mean()
        right = g.loc[g["hemisphere"] == "rh", "vulnerability"].mean()
        seed_rows.append({
            "seed": int(seed),
            "strength_vulnerability_rho": float(rho),
            "mean_vulnerability_left": float(left),
            "mean_vulnerability_right": float(right),
            "right_minus_left": float(right - left),
            "right_over_left": float(right / left),
            "right_greater_than_left": bool(right > left),
        })
    seed_summary = pd.DataFrame(seed_rows).sort_values("seed")
    atomic_write_csv(seed_summary, output_dir / "seed_summary.csv")

    # Regional stability.
    regional_rows = []
    for node, g in results.groupby("node"):
        regional_rows.append({
            "node": int(node),
            "label": str(g["label"].iloc[0]),
            "node_strength": float(g["node_strength"].iloc[0]),
            "mean_vulnerability": float(g["vulnerability"].mean()),
            "sd_vulnerability": float(g["vulnerability"].std(ddof=1))
                if len(g) > 1 else np.nan,
            "median_vulnerability_rank": float(g["vulnerability_rank"].median()),
            "mean_vulnerability_rank": float(g["vulnerability_rank"].mean()),
            "top_n_frequency": float(np.mean(g["vulnerability_rank"] <= top_n)),
        })
    regional = pd.DataFrame(regional_rows).sort_values(
        ["median_vulnerability_rank", "mean_vulnerability_rank"]
    )
    atomic_write_csv(regional, output_dir / "regional_stability.csv")

    # Pairwise seed rank correlations.
    pairwise_rows = []
    if len(completed_seeds) >= 2:
        matrix = results.pivot(
            index="node", columns="seed", values="vulnerability"
        )
        for i, a in enumerate(completed_seeds):
            for b in completed_seeds[i + 1:]:
                rho, _ = spearmanr(matrix[a], matrix[b])
                pairwise_rows.append({
                    "seed_a": int(a),
                    "seed_b": int(b),
                    "vulnerability_rank_rho": float(rho),
                })
    pairwise = pd.DataFrame(pairwise_rows)
    atomic_write_csv(pairwise, output_dir / "pairwise_seed_correlations.csv")

    # Combined regional data for convenient downstream use.
    atomic_write_csv(
        results.sort_values(["seed", "node"]),
        output_dir / "seed_regional_results.csv"
    )

    return results, seed_summary, regional, pairwise


def print_summary(agg, top_n):
    if agg is None:
        return
    results, seed_summary, regional, pairwise = agg
    n_seeds = results["seed"].nunique()

    print("\n" + "=" * 76)
    print(f"SUMMARY FROM {n_seeds} COMPLETED SEEDS")
    print("=" * 76)

    if len(pairwise):
        r = pairwise["vulnerability_rank_rho"]
        print(
            "\nPairwise regional vulnerability rank correlation:"
            f"\n  median rho = {r.median():.4f}"
            f"\n  mean rho   = {r.mean():.4f}"
            f"\n  min rho    = {r.min():.4f}"
            f"\n  max rho    = {r.max():.4f}"
        )

    s = seed_summary["strength_vulnerability_rho"]
    print(
        "\nVulnerability vs structural node strength across seeds:"
        f"\n  median rho = {s.median():.4f}"
        f"\n  range      = {s.min():.4f} to {s.max():.4f}"
    )

    right_fraction = seed_summary["right_greater_than_left"].mean()
    rr = seed_summary["right_over_left"]
    print(
        "\nRight-left asymmetry:"
        f"\n  right mean > left mean in {100*right_fraction:.1f}% of seeds"
        f"\n  median R/L ratio = {rr.median():.4f}"
        f"\n  range R/L ratio  = {rr.min():.4f} to {rr.max():.4f}"
    )

    show = regional.sort_values(
        ["top_n_frequency", "median_vulnerability_rank"],
        ascending=[False, True]
    ).head(15)

    print(f"\nRegions most consistently in top {top_n}:")
    print(show[
        ["label", "mean_vulnerability",
         "median_vulnerability_rank", "top_n_frequency"]
    ].to_string(
        index=False,
        formatters={
            "mean_vulnerability": lambda x: f"{x:.6g}",
            "median_vulnerability_rank": lambda x: f"{x:.1f}",
            "top_n_frequency": lambda x: f"{100*x:.1f}%",
        }
    ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default="data/reference/slow_growth_timescale_model_exact.npz"
    )
    ap.add_argument("--n-seeds", type=int, default=100)
    ap.add_argument("--first-seed", type=int, default=1001)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument(
        "--output-dir",
        default="results/seed_robustness_preadapt"
    )
    args = ap.parse_args()

    out = Path(args.output_dir)
    completed_dir = out / "completed"
    completed_dir.mkdir(parents=True, exist_ok=True)

    # Load only what this analysis requires.
    m = np.load(args.model, allow_pickle=True)
    W0 = m["W0"].astype(float)
    labels = m["labels"].astype(str)
    lags = m["lags"].astype(float)

    n = W0.shape[0]
    node_strength = W0.sum(axis=1)
    target_norm, lesions = matched_lesions(W0)

    seeds = list(range(args.first_seed, args.first_seed + args.n_seeds))

    completed = []
    corrupt = []
    for seed in seeds:
        p = completed_dir / f"seed_{seed}.csv"
        if p.exists():
            if valid_checkpoint(p, seed, n):
                completed.append(seed)
            else:
                corrupt.append(seed)

    for seed in corrupt:
        p = completed_dir / f"seed_{seed}.csv"
        print(f"Removing incomplete/corrupt checkpoint: {p}")
        p.unlink(missing_ok=True)

    todo = [s for s in seeds if s not in set(completed)]

    metadata = {
        "model": str(args.model),
        "n_seeds": args.n_seeds,
        "first_seed": args.first_seed,
        "workers": args.workers,
        "top_n": args.top_n,
        "target_norm": target_norm,
        "n_regions": n,
        "seed_definition": "numpy default_rng(seed); gain then noise_sd; U(0.85,1.15)",
        "analysis": "endpoint pre-adaptation equal-Frobenius focal vulnerability",
    }
    with open(out / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Regions = {n}")
    print(f"Common ||Delta W||F = {target_norm:.10g}")
    print(f"Requested seeds = {len(seeds)}")
    print(f"Completed checkpoints = {len(completed)}")
    print(f"Remaining = {len(todo)}")
    print(f"Workers = {args.workers}")

    if not todo:
        agg = aggregate(completed_dir, seeds, labels, args.top_n, out)
        print_summary(agg, args.top_n)
        print("\nAll requested seeds already complete.")
        return

    t0 = time.time()
    newly_completed = 0

    executor = ProcessPoolExecutor(max_workers=args.workers)
    futures = {}

    try:
        # Submit all remaining seeds. Completed results are checkpointed by
        # the parent process only, avoiding concurrent writes to one file.
        for seed in todo:
            fut = executor.submit(
                run_seed,
                seed,
                W0,
                labels,
                lags,
                lesions,
                node_strength,
            )
            futures[fut] = seed

        for fut in as_completed(futures):
            seed = futures[fut]
            df = fut.result()

            checkpoint = completed_dir / f"seed_{seed}.csv"
            atomic_write_csv(df, checkpoint)
            newly_completed += 1

            elapsed = (time.time() - t0) / 60.0
            total_done = len(completed) + newly_completed

            rho, _ = spearmanr(
                df["node_strength"], df["vulnerability"]
            )
            left = df.loc[
                df["hemisphere"] == "lh", "vulnerability"
            ].mean()
            right = df.loc[
                df["hemisphere"] == "rh", "vulnerability"
            ].mean()

            print(
                f"[{total_done:03d}/{len(seeds):03d}] "
                f"seed {seed} complete | "
                f"strength-V rho={rho:.3f} | "
                f"R/L={right/left:.3f} | "
                f"elapsed={elapsed:.1f} min",
                flush=True,
            )

    except KeyboardInterrupt:
        print(
            "\nInterrupt received. Completed seed checkpoints are preserved.",
            flush=True,
        )

        # Do not wait for unfinished worker jobs. They are intentionally
        # not checkpointed and will be recomputed on restart.
        for fut in futures:
            fut.cancel()

        executor.shutdown(wait=False, cancel_futures=True)

        agg = aggregate(completed_dir, seeds, labels, args.top_n, out)
        print_summary(agg, args.top_n)

        print(
            "\nRun the same command again to resume unfinished seeds.",
            flush=True,
        )
        return

    else:
        executor.shutdown(wait=True)

    agg = aggregate(completed_dir, seeds, labels, args.top_n, out)
    print_summary(agg, args.top_n)

    print("\nSaved:")
    print(out / "seed_regional_results.csv")
    print(out / "seed_summary.csv")
    print(out / "regional_stability.csv")
    print(out / "pairwise_seed_correlations.csv")
    print(f"\nIndividual checkpoints: {completed_dir}")


if __name__ == "__main__":
    main()

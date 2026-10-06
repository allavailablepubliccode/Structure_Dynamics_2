#!/usr/bin/env python3
"""
Parallel, resumable robustness test for FULL focal adaptation.

For each seed:
  - keep the empirical structural connectome W0 fixed;
  - redraw gain_i ~ U(0.85, 1.15);
  - recalibrate global coupling to leading eigenvalue 0.72;
  - redraw noise_sd_i ~ U(0.85, 1.15);
  - recompute the seed-specific baseline CMI target;
  - run the definitive equal-Frobenius focal analysis for all 68 regions.

Each completed seed is saved separately. Re-running the same command skips
completed seeds.

Run from repository root:
  python 08_focal_adaptation_robustness.py --n-seeds 10 --workers 6
"""

# Avoid BLAS oversubscription when using several worker processes.
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import importlib.util
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from model_utils import (
    load_timescale_model,
    gaussian_cmi_vector,
)


def load_definitive_script(path):
    """Load the definitive focal-perturbation analysis as a module."""
    spec = importlib.util.spec_from_file_location(
        "focal06b",
        path,
    )

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return mod


def make_seed_model(base, seed):
    """Construct one gain/noise realization and its baseline CMI target."""

    W0 = base["W0"]
    n = W0.shape[0]

    rng = np.random.default_rng(seed)

    # Exact initialization procedure used for the primary model.
    gain = rng.uniform(
        0.85,
        1.15,
        n,
    )

    rM = np.max(
        np.real(
            np.linalg.eigvals(
                np.diag(gain) @ W0
            )
        )
    )

    coupling = 0.72 / rM

    noise_sd = rng.uniform(
        0.85,
        1.15,
        n,
    )

    theta0 = np.ones(n)

    target = gaussian_cmi_vector(
        W0,
        theta0,
        gain,
        noise_sd,
        coupling,
        base["lags"],
    )

    if target is None:
        raise RuntimeError(
            f"Baseline CMI failed for seed {seed}"
        )

    model = dict(base)

    model["gain"] = gain
    model["noise_sd"] = noise_sd
    model["coupling"] = coupling
    model["target"] = target

    return model


def hemi(label):
    """Return hemisphere from cortical label."""

    s = str(label).lower()

    if s.startswith("lh."):
        return "lh"

    if s.startswith("rh."):
        return "rh"

    return ""


def run_seed(
    seed,
    model_path,
    definitive_script,
    steps,
):
    """Run the complete focal-adaptation analysis for one initialization."""

    definitive = load_definitive_script(
        definitive_script
    )

    base = load_timescale_model(
        model_path
    )

    model = make_seed_model(
        base,
        seed,
    )

    W0 = model["W0"]

    labels = np.asarray(
        model["labels"]
    ).astype(str)

    n = W0.shape[0]

    # Recompute the common equal-Frobenius perturbation magnitude.
    maxnorm = np.array(
        [
            np.linalg.norm(
                definitive.full_incident_lesion(
                    W0,
                    k,
                ),
                "fro",
            )
            for k in range(n)
        ]
    )

    target_norm = float(
        np.min(maxnorm)
    )

    rows = []

    for k in range(n):

        dW, frac, max_feasible = (
            definitive.matched_lesion(
                W0,
                k,
                target_norm,
            )
        )

        df = definitive.run_node(
            model,
            k,
            dW,
            steps,
        )

        g = df.sort_values("s")
        r = g.iloc[-1]

        auc_no = (
            definitive.auc_with_baseline_zero(
                g["s"],
                g["vulnerability"],
            )
        )

        auc_adapt = (
            definitive.auc_with_baseline_zero(
                g["s"],
                g["residual_vulnerability"],
            )
        )

        rows.append(
            {
                "seed": int(seed),
                "node": int(k),
                "label": str(labels[k]),
                "hemisphere": hemi(
                    labels[k]
                ),
                "node_strength": float(
                    W0[k, :].sum()
                ),
                "target_norm": target_norm,
                "incident_fraction_removed": float(
                    frac
                ),
                "max_feasible_perturbation_norm": float(
                    max_feasible
                ),
                "vulnerability": float(
                    r["vulnerability"]
                ),
                "compensability": float(
                    r["fractional_compensation"]
                ),
                "residual_vulnerability": float(
                    r["residual_vulnerability"]
                ),
                "absolute_recovery": float(
                    r["absolute_recovery"]
                ),
                "integrated_vulnerability": float(
                    auc_no
                ),
                "integrated_residual_vulnerability": float(
                    auc_adapt
                ),
                "endpoint_stability_margin": float(
                    r["stability_margin"]
                ),
                "cumulative_theta_change": float(
                    r["cumulative_theta_change"]
                ),
                "coupling": float(
                    model["coupling"]
                ),
            }
        )

    out = pd.DataFrame(rows)

    out["vulnerability_rank"] = (
        out["vulnerability"].rank(
            ascending=False,
            method="average",
        )
    )

    out["residual_rank"] = (
        out["residual_vulnerability"].rank(
            ascending=False,
            method="average",
        )
    )

    return out


def atomic_csv(df, path):
    """Write a CSV atomically."""

    path = Path(path)

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    df.to_csv(
        tmp,
        index=False,
    )

    os.replace(
        tmp,
        path,
    )


def valid_checkpoint(
    path,
    seed,
    n=68,
):
    """Check that a completed seed file is valid."""

    try:

        d = pd.read_csv(path)

        return (
            len(d) == n
            and d["node"].nunique() == n
            and set(
                d["seed"].astype(int)
            ) == {int(seed)}
            and d[
                "residual_vulnerability"
            ].notna().all()
        )

    except Exception:
        return False


def summarize(
    completed_dir,
    seeds,
    output_dir,
    top_n=10,
):
    """Aggregate completed seeds and summarize ranking stability."""

    frames = []

    for seed in seeds:

        p = (
            completed_dir
            / f"seed_{seed}.csv"
        )

        if (
            p.exists()
            and valid_checkpoint(
                p,
                seed,
            )
        ):
            frames.append(
                pd.read_csv(p)
            )

    if not frames:
        return

    d = pd.concat(
        frames,
        ignore_index=True,
    )

    done = sorted(
        d["seed"]
        .astype(int)
        .unique()
    )

    atomic_csv(
        d.sort_values(
            ["seed", "node"]
        ),
        output_dir
        / "seed_regional_results.csv",
    )

    # Pairwise rank correlations across initializations.
    pair_rows = []

    if len(done) > 1:

        vm = d.pivot(
            index="node",
            columns="seed",
            values="vulnerability",
        )

        rm = d.pivot(
            index="node",
            columns="seed",
            values="residual_vulnerability",
        )

        for i, a in enumerate(done):

            for b in done[i + 1:]:

                rv, _ = spearmanr(
                    vm[a],
                    vm[b],
                )

                rr, _ = spearmanr(
                    rm[a],
                    rm[b],
                )

                pair_rows.append(
                    {
                        "seed_a": int(a),
                        "seed_b": int(b),
                        "vulnerability_rank_rho": float(
                            rv
                        ),
                        "residual_rank_rho": float(
                            rr
                        ),
                    }
                )

    pairwise = pd.DataFrame(
        pair_rows
    )

    atomic_csv(
        pairwise,
        output_dir
        / "pairwise_seed_rank_correlations.csv",
    )

    # Regional stability across seeds.
    regional_rows = []

    for node, g in d.groupby("node"):

        regional_rows.append(
            {
                "node": int(node),
                "label": g[
                    "label"
                ].iloc[0],
                "mean_vulnerability": (
                    g["vulnerability"].mean()
                ),
                "median_vulnerability_rank": (
                    g[
                        "vulnerability_rank"
                    ].median()
                ),
                "initial_top10_frequency": (
                    np.mean(
                        g[
                            "vulnerability_rank"
                        ]
                        <= top_n
                    )
                ),
                "mean_residual_vulnerability": (
                    g[
                        "residual_vulnerability"
                    ].mean()
                ),
                "median_residual_rank": (
                    g[
                        "residual_rank"
                    ].median()
                ),
                "residual_top10_frequency": (
                    np.mean(
                        g[
                            "residual_rank"
                        ]
                        <= top_n
                    )
                ),
                "mean_compensability": (
                    g[
                        "compensability"
                    ].mean()
                ),
            }
        )

    regional = pd.DataFrame(
        regional_rows
    )

    atomic_csv(
        regional,
        output_dir
        / "regional_stability.csv",
    )

    # Per-seed summaries.
    seed_rows = []

    for seed, g in d.groupby("seed"):

        left = g.loc[
            g["hemisphere"] == "lh",
            "residual_vulnerability",
        ].mean()

        right = g.loc[
            g["hemisphere"] == "rh",
            "residual_vulnerability",
        ].mean()

        rho_strength, _ = spearmanr(
            g["node_strength"],
            g["residual_vulnerability"],
        )

        seed_rows.append(
            {
                "seed": int(seed),
                "residual_strength_rho": float(
                    rho_strength
                ),
                "mean_residual_left": float(
                    left
                ),
                "mean_residual_right": float(
                    right
                ),
                "residual_right_over_left": float(
                    right / left
                ),
                "mean_compensability": float(
                    g[
                        "compensability"
                    ].mean()
                ),
            }
        )

    seed_summary = pd.DataFrame(
        seed_rows
    )

    atomic_csv(
        seed_summary,
        output_dir
        / "seed_summary.csv",
    )

    print(
        "\n"
        + "=" * 76
    )

    print(
        f"SUMMARY FROM {len(done)} "
        "COMPLETED FULL-ADAPTATION SEEDS"
    )

    print(
        "=" * 76
    )

    if len(pairwise):

        x = pairwise[
            "vulnerability_rank_rho"
        ]

        y = pairwise[
            "residual_rank_rho"
        ]

        print(
            "\nInitial vulnerability pairwise "
            "rank correlation:"
            f"\n  median rho = "
            f"{x.median():.4f}"
            f"\n  range      = "
            f"{x.min():.4f} to "
            f"{x.max():.4f}"
        )

        print(
            "\nResidual vulnerability pairwise "
            "rank correlation:"
            f"\n  median rho = "
            f"{y.median():.4f}"
            f"\n  range      = "
            f"{y.min():.4f} to "
            f"{y.max():.4f}"
        )

    show = (
        regional.sort_values(
            [
                "residual_top10_frequency",
                "median_residual_rank",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .head(15)
    )

    print(
        "\nRegions most consistently "
        "in residual top 10:"
    )

    print(
        show[
            [
                "label",
                "mean_residual_vulnerability",
                "median_residual_rank",
                "residual_top10_frequency",
                "mean_compensability",
            ]
        ].to_string(
            index=False,
            formatters={
                "mean_residual_vulnerability":
                    lambda x: f"{x:.6g}",
                "median_residual_rank":
                    lambda x: f"{x:.1f}",
                "residual_top10_frequency":
                    lambda x: f"{100*x:.1f}%",
                "mean_compensability":
                    lambda x: f"{x:.3f}",
            },
        )
    )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--model",
        default=(
            "data/generated/"
            "slow_growth_timescale_model_exact.npz"
        ),
    )

    ap.add_argument(
        "--definitive-script",
        default=(
            "06b_definitive_targeted_lesions.py"
        ),
    )

    ap.add_argument(
        "--n-seeds",
        type=int,
        default=10,
    )

    ap.add_argument(
        "--first-seed",
        type=int,
        default=2001,
    )

    ap.add_argument(
        "--workers",
        type=int,
        default=6,
    )

    ap.add_argument(
        "--steps",
        type=int,
        default=20,
    )

    ap.add_argument(
        "--output-dir",
        default=(
            "results/"
            "seed_robustness_adaptation"
        ),
    )

    args = ap.parse_args()

    out = Path(
        args.output_dir
    )

    completed_dir = (
        out / "completed"
    )

    completed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    seeds = list(
        range(
            args.first_seed,
            args.first_seed
            + args.n_seeds,
        )
    )

    completed = []

    for seed in seeds:

        p = (
            completed_dir
            / f"seed_{seed}.csv"
        )

        if p.exists():

            if valid_checkpoint(
                p,
                seed,
            ):
                completed.append(
                    seed
                )

            else:

                print(
                    "Removing incomplete "
                    f"checkpoint: {p}"
                )

                p.unlink()

    completed_set = set(
        completed
    )

    todo = [
        s
        for s in seeds
        if s not in completed_set
    ]

    print(
        f"Requested seeds = "
        f"{len(seeds)}"
    )

    print(
        f"Completed = "
        f"{len(completed)}"
    )

    print(
        f"Remaining = "
        f"{len(todo)}"
    )

    print(
        f"Workers = "
        f"{args.workers}"
    )

    print(
        f"Adaptive steps = "
        f"{args.steps}"
    )

    if not todo:

        summarize(
            completed_dir,
            seeds,
            out,
        )

        return

    t0 = time.time()

    executor = ProcessPoolExecutor(
        max_workers=args.workers
    )

    futures = {}

    try:

        for seed in todo:

            fut = executor.submit(
                run_seed,
                seed,
                args.model,
                args.definitive_script,
                args.steps,
            )

            futures[fut] = seed

        newly_done = 0

        for fut in as_completed(
            futures
        ):

            seed = futures[fut]
            df = fut.result()

            atomic_csv(
                df,
                completed_dir
                / f"seed_{seed}.csv",
            )

            newly_done += 1

            total = (
                len(completed)
                + newly_done
            )

            elapsed = (
                time.time() - t0
            ) / 60.0

            print(
                f"[{total:02d}/"
                f"{len(seeds):02d}] "
                f"seed {seed} complete | "
                f"elapsed={elapsed:.1f} min",
                flush=True,
            )

    except KeyboardInterrupt:

        print(
            "\nInterrupt received. "
            "Completed seed files are "
            "preserved.",
            flush=True,
        )

        for fut in futures:
            fut.cancel()

        executor.shutdown(
            wait=False,
            cancel_futures=True,
        )

        summarize(
            completed_dir,
            seeds,
            out,
        )

        print(
            "\nRun the same command again "
            "to resume.",
            flush=True,
        )

        return

    else:

        executor.shutdown(
            wait=True
        )

    summarize(
        completed_dir,
        seeds,
        out,
    )

    print(
        "\nSaved under:"
    )

    print(out)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Robustness analysis across alternative patterns of distributed structural change."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from model_utils import (
    load_timescale_model,
    gaussian_cmi_vector,
    normalized_error,
    theta_jacobian,
    safe_backtrack,
    matching_index_binary,
    JAC_REFRESH,
)


N_LIST = [5, 10, 20, 40]


def norm_to(X, ref):
    """Scale X to have the same Frobenius norm as ref."""
    n = np.linalg.norm(X, "fro")

    if n == 0:
        raise ValueError("Cannot normalize a zero structural-change matrix.")

    return X * (
        np.linalg.norm(ref, "fro") / n
    )


def build_patterns(model, struct):
    """Construct the four structural-change patterns used in the manuscript."""

    W0 = model["W0"]

    primary = model["DeltaW"].copy()

    D = struct["D"].astype(float)
    M = struct["M"].astype(float)
    dmed = float(struct["dmed"])
    mu = float(struct["matching_offset"])

    mask = (W0 > 0).astype(float)
    np.fill_diagonal(mask, 0.0)

    patterns = {
        "Primary": primary,

        "Proportional": norm_to(
            W0 * mask,
            primary,
        ),

        "Distance-dependent": norm_to(
            W0 * np.exp(-D / dmed) * mask,
            primary,
        ),

        "Matching-index-dependent": norm_to(
            W0 * (M + mu) * mask,
            primary,
        ),
    }

    # Preserve an undirected structural network and zero diagonal.
    for name in patterns:
        X = patterns[name]
        X = 0.5 * (X + X.T)
        np.fill_diagonal(X, 0.0)
        patterns[name] = X

    return patterns


def run_condition(model, DeltaW, N):
    """Run one structural-change pattern for a specified number of increments."""

    W0 = model["W0"]
    gain = model["gain"]
    noise = model["noise_sd"]
    coupling = model["coupling"]
    target = model["target"]
    lags = model["lags"]

    theta = np.ones(W0.shape[0])

    J = None
    rows = []

    for step in range(1, N + 1):

        s = step / N
        W = W0 + s * DeltaW

        C_before = gaussian_cmi_vector(
            W,
            theta,
            gain,
            noise,
            coupling,
            lags,
        )

        if C_before is None:
            raise RuntimeError(
                f"Unstable model before adaptation at "
                f"N={N}, step={step}"
            )

        if (
            J is None
            or step == 1
            or (step - 1) % JAC_REFRESH == 0
        ):
            J = theta_jacobian(
                W,
                theta,
                gain,
                noise,
                coupling,
                lags,
            )

        if J is None:
            dtheta = np.zeros_like(theta)
            C_after = C_before

        else:
            residual = C_before - target

            dtheta = -np.linalg.lstsq(
                J,
                residual,
                rcond=None,
            )[0]

            # Per-parameter adaptation bound.
            dtheta = np.clip(
                dtheta,
                -model["per_node"],
                model["per_node"],
            )

            # Total L2 adaptation bound.
            dn = np.linalg.norm(dtheta)

            if dn > model["B_step"] and dn > 0:
                dtheta *= model["B_step"] / dn

            dtheta, C_after = safe_backtrack(
                W,
                theta,
                dtheta,
                gain,
                noise,
                coupling,
                lags,
            )

        theta = theta + dtheta

        err = normalized_error(
            C_after,
            target,
        )

        rows.append(
            {
                "N": N,
                "step": step,
                "s": s,
                "error": err,
            }
        )

    return pd.DataFrame(rows)


def integrated_error(df):
    """Area under normalized CMI error as a function of structural progression."""

    g = df.sort_values("s")

    x = g["s"].to_numpy()
    y = g["error"].to_numpy()

    # Include the baseline point E=0 at s=0.
    x = np.concatenate(([0.0], x))
    y = np.concatenate(([0.0], y))

    return float(
        np.trapezoid(y, x)
    )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--model",
        default="data/generated/slow_growth_timescale_model_exact.npz",
    )

    ap.add_argument(
        "--structural-model",
        default="data/generated/slow_growth_structural_model.npz",
    )

    ap.add_argument(
        "--output-dir",
        default="results/generated/structural_pattern_robustness",
    )

    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = load_timescale_model(args.model)

    struct = np.load(
        args.structural_model,
        allow_pickle=True,
    )

    patterns = build_patterns(
        model,
        struct,
    )

    trajectory_frames = []
    summaries = []

    for pattern_name, DeltaW in patterns.items():

        print(f"\n=== {pattern_name} ===")

        for N in N_LIST:

            print(f"Running N={N}")

            df = run_condition(
                model,
                DeltaW,
                N,
            )

            df.insert(
                0,
                "pattern",
                pattern_name,
            )

            trajectory_frames.append(df)

            area = integrated_error(df)

            endpoint = float(
                df.iloc[-1]["error"]
            )

            summaries.append(
                {
                    "pattern": pattern_name,
                    "N": N,
                    "integrated_error": area,
                    "endpoint_error": endpoint,
                }
            )

            print(
                f"N={N:2d}: "
                f"endpoint E={endpoint:.6f}, "
                f"integrated error={area:.6f}"
            )

    trajectories = pd.concat(
        trajectory_frames,
        ignore_index=True,
    )

    summary = pd.DataFrame(
        summaries
    )

    trajectories.to_csv(
        out / "structural_pattern_trajectories.csv",
        index=False,
    )

    summary.to_csv(
        out / "structural_pattern_summary.csv",
        index=False,
    )

    print("\nFINAL SUMMARY")
    print(
        summary.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
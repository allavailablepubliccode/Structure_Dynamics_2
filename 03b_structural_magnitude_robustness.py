#!/usr/bin/env python3
"""
Robustness of the primary distributed structural-change analysis
to the overall magnitude of structural change.

Tests mean relative increases of 5%, 10%, and 20%.
The original model DeltaW corresponds to 20%, so only DeltaW
is rescaled; all dynamical/adaptation parameters are unchanged.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from model_utils import (
    load_timescale_model,
    gaussian_cmi_vector,
    normalized_error,
    theta_jacobian,
    safe_backtrack,
    drift,
    max_real_eig,
    B_STEP,
    JAC_REFRESH,
)


N_LIST = [5, 10, 20, 40]

# Original DeltaW corresponds to mean 20% increase.
GROWTH_AMPS = [0.05, 0.10, 0.20]
ORIGINAL_GROWTH_AMP = 0.20


def run_condition(model, N, growth_amp):

    W0 = model["W0"]

    # Rescale ONLY the structural perturbation magnitude.
    scale = growth_amp / ORIGINAL_GROWTH_AMP
    dW = model["DeltaW"] * scale

    gain = model["gain"]
    noise = model["noise_sd"]
    c = model["coupling"]
    target = model["target"]
    lags = model["lags"]
    per_node = model["per_node"]

    theta0 = np.ones(W0.shape[0])
    theta = theta0.copy()
    J = None
    rows = []

    for step in range(1, N + 1):

        s = step / N
        W = W0 + s * dW

        # Error without adaptation
        C_no = gaussian_cmi_vector(
            W, theta0, gain, noise, c, lags
        )
        err_no = normalized_error(C_no, target)

        # Current adapted state
        C_before = gaussian_cmi_vector(
            W, theta, gain, noise, c, lags
        )

        if C_before is None:

            dtheta = np.ones_like(theta)
            dtheta *= B_STEP / np.linalg.norm(dtheta)

            dtheta, C_after = safe_backtrack(
                W, theta, dtheta,
                gain, noise, c, lags
            )
            J = None

        else:

            if (
                J is None
                or step == 1
                or (step - 1) % JAC_REFRESH == 0
            ):
                J = theta_jacobian(
                    W, theta, gain, noise, c, lags
                )

            if J is None:

                dtheta = np.zeros_like(theta)
                C_after = C_before

            else:

                sol = lsq_linear(
                    J,
                    -(C_before - target),
                    bounds=(-per_node, per_node),
                    lsmr_tol="auto",
                    max_iter=200,
                )

                dtheta = sol.x
                dn = np.linalg.norm(dtheta)

                if dn > B_STEP and dn > 0:
                    dtheta *= B_STEP / dn

                dtheta, C_after = safe_backtrack(
                    W, theta, dtheta,
                    gain, noise, c, lags
                )

        theta = theta + dtheta

        err = normalized_error(C_after, target)

        K = (
            1 - err / err_no
            if err_no > 1e-15
            else np.nan
        )

        margin = -max_real_eig(
            drift(W, theta, gain, c)
        )

        rows.append(
            dict(
                growth_amp=growth_amp,
                n_steps=N,
                step=step,
                s=s,
                error_nocomp=err_no,
                error_comp=err,
                K=K,
                stability_margin=margin,
                dtheta_norm=np.linalg.norm(dtheta),
                theta_norm=np.linalg.norm(theta - theta0),
            )
        )

        print(
            f"growth={100*growth_amp:4.0f}% "
            f"N={N:2d} "
            f"step={step:2d}/{N}: "
            f"E_no={err_no:.6f}, "
            f"E={err:.6f}, "
            f"K={K:.6f}, "
            f"m={margin:.6f}"
        )

    return pd.DataFrame(rows)


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--model",
        default="data/generated/"
                "slow_growth_timescale_model_exact.npz",
    )

    ap.add_argument(
        "--output-dir",
        default="results/generated/"
                "growth_magnitude_robustness",
    )

    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = load_timescale_model(args.model)

    # --------------------------------------------------------
    # Verify that the original DeltaW really corresponds
    # to a mean 20% relative increase over existing edges.
    # --------------------------------------------------------

    W0 = model["W0"]
    dW_original = model["DeltaW"]

    mask = W0 > 0

    relative_change = (
        dW_original[mask] / W0[mask]
    )

    print(
        "\nOriginal DeltaW:"
        f"\nMean relative increase = "
        f"{relative_change.mean():.6f}"
        f"\nMedian relative increase = "
        f"{np.median(relative_change):.6f}"
        "\n"
    )

    # --------------------------------------------------------
    # Run all conditions
    # --------------------------------------------------------

    dfs = []

    for growth_amp in GROWTH_AMPS:

        print("\n" + "=" * 60)
        print(
            f"MEAN STRUCTURAL INCREASE: "
            f"{100 * growth_amp:.0f}%"
        )
        print("=" * 60)

        for N in N_LIST:

            dfs.append(
                run_condition(
                    model,
                    N,
                    growth_amp,
                )
            )

    all_df = pd.concat(
        dfs,
        ignore_index=True,
    )

    all_df.to_csv(
        out / "growth_magnitude_trajectories.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Summaries
    # --------------------------------------------------------

    summaries = []

    for (growth_amp, N), g in all_df.groupby(
        ["growth_amp", "n_steps"]
    ):

        g = g.sort_values("s")

        x = g.s.to_numpy()
        y = g.error_comp.to_numpy()

        integrated = float(
            np.trapezoid(y, x)
        )

        r = g.iloc[-1]

        summaries.append(
            dict(
                growth_amp=growth_amp,
                mean_structural_increase_percent=
                    100 * growth_amp,
                n_steps=N,
                endpoint_error_nocomp=
                    r.error_nocomp,
                endpoint_error=
                    r.error_comp,
                fractional_compensation=
                    r.K,
                integrated_error=
                    integrated,
                stability_margin=
                    r.stability_margin,
            )
        )

    summary_df = pd.DataFrame(summaries)

    summary_df = summary_df.sort_values(
        ["growth_amp", "n_steps"]
    )

    summary_df.to_csv(
        out / "growth_magnitude_summary.csv",
        index=False,
    )

    print("\n\nFINAL SUMMARY")
    print("=" * 90)

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )


if __name__ == "__main__":
    main()
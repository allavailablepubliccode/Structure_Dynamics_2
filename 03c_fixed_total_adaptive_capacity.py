#!/usr/bin/env python3
"""
Fixed-total-adaptive-capacity control.

Runs N = 5, 10, 20, 40 while holding total permitted adaptation
constant across conditions.

The total budget is matched to the standard N=10 condition:
    N=5:  B_step=0.024, per_node=0.008
    N=10: B_step=0.012, per_node=0.004
    N=20: B_step=0.006, per_node=0.002
    N=40: B_step=0.003, per_node=0.001
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
    JAC_REFRESH,
)

N_LIST = [5, 10, 20, 40]

# Standard per-step bounds in the main analysis
BASE_N = 10
BASE_B_STEP = 0.012
BASE_PER_NODE = 0.004


def run_condition(model, N):

    W0 = model["W0"]
    dW = model["DeltaW"]
    gain = model["gain"]
    noise = model["noise_sd"]
    c = model["coupling"]
    target = model["target"]
    lags = model["lags"]

    # Scale per-step adaptation inversely with N so that
    # total permitted adaptation is identical for every condition.
    scale = BASE_N / N
    B_step = BASE_B_STEP * scale
    per_node = BASE_PER_NODE * scale

    print(
        f"\nN={N}: "
        f"B_step={B_step:.6f}, "
        f"per_node={per_node:.6f}, "
        f"total L2 budget={N * B_step:.6f}"
    )

    theta0 = np.ones(W0.shape[0])
    theta = theta0.copy()
    J = None
    rows = []

    for step in range(1, N + 1):

        s = step / N
        W = W0 + s * dW

        # No-adaptation reference
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
            dtheta *= B_step / np.linalg.norm(dtheta)

            dtheta, C_after = safe_backtrack(
                W, theta, dtheta, gain, noise, c, lags
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

                if dn > B_step and dn > 0:
                    dtheta *= B_step / dn

                dtheta, C_after = safe_backtrack(
                    W, theta, dtheta, gain, noise, c, lags
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
                n_steps=N,
                step=step,
                s=s,
                B_step=B_step,
                per_node=per_node,
                error_nocomp=err_no,
                error_comp=err,
                K=K,
                stability_margin=margin,
                dtheta_norm=np.linalg.norm(dtheta),
                theta_norm=np.linalg.norm(theta - theta0),
            )
        )

        print(
            f"N={N:2d} "
            f"step={step:2d}/{N}: "
            f"E={err:.6f}, "
            f"K={K:.6f}, "
            f"m={margin:.6f}"
        )

    return pd.DataFrame(rows)


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--model",
        default="data/generated/slow_growth_timescale_model_exact.npz",
    )

    ap.add_argument(
        "--output-dir",
        default="results/generated/fixed_total_adaptive_capacity",
    )

    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = load_timescale_model(args.model)

    all_df = pd.concat(
        [run_condition(model, N) for N in N_LIST],
        ignore_index=True,
    )

    all_df.to_csv(
        out / "fixed_capacity_trajectories.csv",
        index=False,
    )

    summaries = []

    for N, g in all_df.groupby("n_steps"):

        g = g.sort_values("s")

        x = g.s.to_numpy()
        y = g.error_comp.to_numpy()

        integrated = float(
            np.trapezoid(y, x)
        )

        r = g.iloc[-1]

        summaries.append(
            dict(
                n_steps=N,
                B_step=r.B_step,
                per_node=r.per_node,
                endpoint_error_nocomp=r.error_nocomp,
                endpoint_error=r.error_comp,
                fractional_compensation=r.K,
                integrated_error=integrated,
                stability_margin=r.stability_margin,
            )
        )

    summary = pd.DataFrame(summaries)

    summary.to_csv(
        out / "fixed_capacity_summary.csv",
        index=False,
    )

    print("\nFINAL SUMMARY")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
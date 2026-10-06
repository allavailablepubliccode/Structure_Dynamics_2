#!/usr/bin/env python3
"""
Proportional focal-perturbation control.

Control question
----------------
Does regional heterogeneity persist when every cortical node loses the same
fraction of its incident structural connectivity, rather than the same
absolute Frobenius magnitude?

The default fraction is 0.10 (10%). This is a distinct normalization/control,
not a model of literal 10% tissue loss.

Uses the same OU/CMI/adaptation machinery as the main manuscript analysis and
the definitive equal-magnitude focal-perturbation analysis.

Run from repository root:
 python 06c_proportional_lesion_control.py \
   --model data/generated/slow_growth_timescale_model_exact.npz \
   --fraction 0.10 \
   --steps 20 \
   --output-dir results/targeted_lesions_proportional

Safe to interrupt with Ctrl+C. Re-running resumes from completed nodes.
"""

import argparse
import time
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


def proportional_lesion(W0, k, fraction):
    """Remove a fixed fraction of all connections incident on node k."""

    dW = np.zeros_like(W0, dtype=float)
    dW[k, :] = -fraction * W0[k, :]
    dW[:, k] = -fraction * W0[:, k]
    dW[k, k] = 0.0

    return dW


def run_node(model, node, dW, N):
    """Run one proportional focal perturbation with and without adaptation."""

    W0 = model["W0"]
    gain = model["gain"]
    noise = model["noise_sd"]
    c = model["coupling"]
    target = model["target"]
    lags = model["lags"]

    B_step = model["B_step"]
    per_node = model["per_node"]

    theta0 = np.ones(W0.shape[0])
    theta = theta0.copy()

    J = None
    rows = []

    for step in range(1, N + 1):

        s = step / N
        W = W0 + s * dW

        # Structural perturbation without dynamical adaptation.
        C_no = gaussian_cmi_vector(
            W,
            theta0,
            gain,
            noise,
            c,
            lags,
        )

        err_no = normalized_error(
            C_no,
            target,
        )

        # Structural perturbation with dynamical adaptation.
        C_before = gaussian_cmi_vector(
            W,
            theta,
            gain,
            noise,
            c,
            lags,
        )

        if C_before is None:

            dtheta = np.ones_like(theta)

            dtheta *= (
                B_step
                / np.linalg.norm(dtheta)
            )

            dtheta, C_after = safe_backtrack(
                W,
                theta,
                dtheta,
                gain,
                noise,
                c,
                lags,
            )

            J = None

        else:

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
                    c,
                    lags,
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
                    W,
                    theta,
                    dtheta,
                    gain,
                    noise,
                    c,
                    lags,
                )

        theta += dtheta

        err_adapt = normalized_error(
            C_after,
            target,
        )

        absolute_recovery = (
            err_no - err_adapt
        )

        K = (
            1 - err_adapt / err_no
            if err_no > 1e-12
            else np.nan
        )

        margin = -max_real_eig(
            drift(
                W,
                theta,
                gain,
                c,
            )
        )

        rows.append(
            dict(
                node=node,
                step=step,
                s=s,
                vulnerability=err_no,
                residual_vulnerability=err_adapt,
                absolute_recovery=absolute_recovery,
                fractional_compensation=K,
                stability_margin=margin,
                dtheta_norm=np.linalg.norm(dtheta),
                cumulative_theta_change=np.linalg.norm(
                    theta - theta0
                ),
            )
        )

    return pd.DataFrame(rows)


def auc_with_baseline_zero(s, y):
    """Integrate CMI error over structural progression, including s=0."""

    x = np.concatenate(
        ([0.0], np.asarray(s, float))
    )

    z = np.concatenate(
        ([0.0], np.asarray(y, float))
    )

    return float(
        np.trapezoid(z, x)
    )


def main():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--model",
        default=(
            "data/generated/"
            "slow_growth_timescale_model_exact.npz"
        ),
    )

    p.add_argument(
        "--fraction",
        type=float,
        default=0.10,
    )

    p.add_argument(
        "--steps",
        type=int,
        default=20,
    )

    p.add_argument(
        "--output-dir",
        default="results/targeted_lesions_proportional",
    )

    p.add_argument(
        "--nodes",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Optional zero-based subset of nodes "
            "for testing."
        ),
    )

    args = p.parse_args()

    if not 0 < args.fraction <= 1:
        raise ValueError(
            "--fraction must be in (0, 1]."
        )

    out = Path(args.output_dir)
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = load_timescale_model(
        args.model
    )

    W0 = model["W0"]
    labels = model["labels"]
    n = W0.shape[0]

    requested = (
        list(range(n))
        if args.nodes is None
        else args.nodes
    )

    summary_path = (
        out / "focal_perturbation_summary.csv"
    )

    traj_path = (
        out / "focal_perturbation_trajectories.csv"
    )

    # Resume completed analyses if output already exists.
    if summary_path.exists():

        old_summary = pd.read_csv(
            summary_path
        )

        if len(old_summary):

            if not np.allclose(
                old_summary[
                    "incident_fraction_removed"
                ].astype(float),
                args.fraction,
            ):
                raise ValueError(
                    "Existing output uses a different "
                    "perturbation fraction."
                )

            if not np.all(
                old_summary[
                    "n_steps"
                ].astype(int)
                == args.steps
            ):
                raise ValueError(
                    "Existing output uses a different "
                    "--steps value."
                )

        completed = set(
            old_summary.node.astype(int)
        )

        summaries = (
            old_summary.to_dict("records")
        )

    else:

        completed = set()
        summaries = []

    if traj_path.exists():

        old_traj = pd.read_csv(
            traj_path
        )

        old_traj = old_traj[
            old_traj.node.astype(int).isin(
                completed
            )
        ]

        trajectories = (
            [old_traj]
            if len(old_traj)
            else []
        )

    else:

        trajectories = []

    todo = [
        k
        for k in requested
        if k not in completed
    ]

    print(
        f"Proportional focal perturbation: "
        f"{100 * args.fraction:.1f}% "
        f"incident connectivity; "
        f"steps={args.steps}; "
        f"remaining={len(todo)}"
    )

    if not todo:
        print(
            "All requested nodes already complete."
        )
        return

    t0 = time.time()

    for i, k in enumerate(
        todo,
        1,
    ):

        dW = proportional_lesion(
            W0,
            k,
            args.fraction,
        )

        perturbation_norm = float(
            np.linalg.norm(
                dW,
                "fro",
            )
        )

        print(
            f"[{i:02d}/{len(todo):02d}] "
            f"{k:02d} {labels[k]} | "
            f"||dW||F="
            f"{perturbation_norm:.6g}"
        )

        df = run_node(
            model,
            k,
            dW,
            args.steps,
        )

        df.insert(
            1,
            "label",
            labels[k],
        )

        df[
            "incident_fraction_removed"
        ] = args.fraction

        df[
            "perturbation_norm"
        ] = perturbation_norm

        trajectories.append(df)

        g = df.sort_values("s")
        r = g.iloc[-1]

        auc_no = auc_with_baseline_zero(
            g.s,
            g.vulnerability,
        )

        auc_adapt = (
            auc_with_baseline_zero(
                g.s,
                g.residual_vulnerability,
            )
        )

        summaries.append(
            dict(
                node=k,
                label=labels[k],
                n_steps=args.steps,
                incident_fraction_removed=(
                    args.fraction
                ),
                perturbation_norm=(
                    perturbation_norm
                ),
                node_strength=float(
                    W0[k, :].sum()
                ),
                node_degree=int(
                    np.count_nonzero(
                        W0[k, :]
                    )
                ),
                vulnerability=float(
                    r.vulnerability
                ),
                compensability=float(
                    r.fractional_compensation
                ),
                residual_vulnerability=float(
                    r.residual_vulnerability
                ),
                absolute_recovery=float(
                    r.absolute_recovery
                ),
                integrated_vulnerability=(
                    auc_no
                ),
                integrated_residual_vulnerability=(
                    auc_adapt
                ),
                integrated_absolute_recovery=(
                    auc_no - auc_adapt
                ),
                endpoint_stability_margin=float(
                    r.stability_margin
                ),
                cumulative_theta_change=float(
                    r.cumulative_theta_change
                ),
            )
        )

        pd.concat(
            trajectories,
            ignore_index=True,
        ).to_csv(
            traj_path,
            index=False,
        )

        pd.DataFrame(
            summaries
        ).to_csv(
            summary_path,
            index=False,
        )

        elapsed = (
            time.time() - t0
        ) / 60

        print(
            f"    V={r.vulnerability:.6g} | "
            f"K={r.fractional_compensation:.3f} | "
            f"R={r.residual_vulnerability:.6g} | "
            f"elapsed={elapsed:.1f} min"
        )

    summary = pd.DataFrame(
        summaries
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    summary.sort_values(
        "vulnerability",
        ascending=False,
    ).to_csv(
        out / "ranked_vulnerability.csv",
        index=False,
    )

    summary.sort_values(
        "residual_vulnerability",
        ascending=False,
    ).to_csv(
        out
        / "ranked_residual_vulnerability.csv",
        index=False,
    )

    summary.sort_values(
        "compensability",
        ascending=False,
    ).to_csv(
        out / "ranked_compensability.csv",
        index=False,
    )

    print(
        "\nProportional focal-perturbation "
        "control analysis complete."
    )


if __name__ == "__main__":
    main()
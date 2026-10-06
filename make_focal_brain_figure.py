#!/usr/bin/env python3
"""
Generate cortical maps of focal vulnerability and residual vulnerability.

The script places the 68 Desikan-Killiany cortical regions at their parcel
centroids on the FreeSurfer fsaverage pial surface.

Initial and residual vulnerability are displayed using the same quantitative
colour scale so that the two maps are directly comparable.

Requirements
------------
A local FreeSurfer fsaverage directory must be available at:

    fsaverage_for_figure/

containing at least:

    surf/lh.pial
    surf/rh.pial
    label/lh.aparc.annot
    label/rh.aparc.annot

The fsaverage files themselves are not included in this repository.

Input
-----
results/targeted_lesions_definitive/focal_perturbation_summary.csv

Output
------
focal_figure_output/
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import cm, colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from nibabel.freesurfer.io import read_geometry, read_annot


# ============================================================
# SETTINGS
# ============================================================

HERE = Path(__file__).resolve().parent

RESULTS_FILE = (
    HERE
    / "results"
    / "targeted_lesions_definitive"
    / "focal_perturbation_summary.csv"
)

FSAVERAGE = (
    HERE
    / "fsaverage_for_figure"
)

OUTPUT = (
    HERE
    / "focal_figure_output"
)

OUTPUT.mkdir(
    exist_ok=True
)

# Full fsaverage mesh. Do not randomly subsample faces.
SURFACE_ALPHA = 0.08
NODE_SIZE = 65

# Initial and residual vulnerability use the same quantitative scale.
CMAP = "viridis"

VIEWS = {
    "left_lateral": dict(
        elev=8,
        azim=180,
    ),
    "superior": dict(
        elev=90,
        azim=-90,
    ),
    "posterior": dict(
        elev=8,
        azim=90,
    ),
    "right_lateral": dict(
        elev=8,
        azim=0,
    ),
}


# ============================================================
# LOAD FREESURFER DATA
# ============================================================

surface = {}
parcel_centres = []

for hemi in [
    "lh",
    "rh",
]:

    vertices, faces = read_geometry(
        FSAVERAGE
        / "surf"
        / f"{hemi}.pial"
    )

    (
        annot_labels,
        ctab,
        names,
    ) = read_annot(
        FSAVERAGE
        / "label"
        / f"{hemi}.aparc.annot"
    )

    names = [
        (
            n.decode("utf-8")
            if isinstance(n, bytes)
            else str(n)
        )
        for n in names
    ]

    surface[hemi] = {
        "vertices": vertices,
        "faces": faces,
    }

    # Compute the centroid of each Desikan-Killiany parcel.
    for parcel_index, parcel_name in enumerate(
        names
    ):

        if parcel_name in {
            "unknown",
            "corpuscallosum",
        }:
            continue

        vertex_indices = np.where(
            annot_labels == parcel_index
        )[0]

        if len(vertex_indices) == 0:
            continue

        centre = vertices[
            vertex_indices
        ].mean(
            axis=0
        )

        parcel_centres.append(
            {
                "label": (
                    f"{hemi}."
                    f"{parcel_name}"
                ),
                "x": centre[0],
                "y": centre[1],
                "z": centre[2],
            }
        )


centres = pd.DataFrame(
    parcel_centres
)


# ============================================================
# LOAD FOCAL-PERTURBATION RESULTS
# ============================================================

results = pd.read_csv(
    RESULTS_FILE
)

required = {
    "label",
    "vulnerability",
    "residual_vulnerability",
}

missing = (
    required
    - set(results.columns)
)

if missing:
    raise ValueError(
        "Missing columns from "
        "focal_perturbation_summary.csv: "
        f"{missing}"
    )


data = results.merge(
    centres,
    on="label",
    how="left",
    validate="one_to_one",
)


if data[
    ["x", "y", "z"]
].isna().any().any():

    missing_regions = data.loc[
        data["x"].isna(),
        "label",
    ].tolist()

    raise RuntimeError(
        "Could not match these regions "
        "to fsaverage:\n"
        + "\n".join(
            missing_regions
        )
    )


if len(data) != 68:
    raise RuntimeError(
        "Expected 68 cortical regions "
        f"but matched {len(data)}."
    )


print(
    "Successfully matched all 68 "
    "empirical cortical regions."
)


# ============================================================
# COLOUR SCALE
# ============================================================

# Initial and residual vulnerability share exactly the same scale.
# The upper limit is determined by the largest initial vulnerability.
vmax = data[
    "vulnerability"
].max()

norm = colors.Normalize(
    vmin=0,
    vmax=vmax,
)

cmap = plt.get_cmap(
    CMAP
)


# ============================================================
# DRAWING FUNCTIONS
# ============================================================

def add_brain(ax):
    """
    Draw intact fsaverage pial surfaces.

    No face deletion or random triangle subsampling is performed.
    """

    for hemi in [
        "lh",
        "rh",
    ]:

        vertices = (
            surface[hemi][
                "vertices"
            ]
        )

        faces = (
            surface[hemi][
                "faces"
            ]
        )

        brain = Poly3DCollection(
            vertices[faces],
            linewidth=0,
            edgecolor="none",
            alpha=SURFACE_ALPHA,
        )

        # Neutral anatomical background.
        brain.set_facecolor(
            "lightgray"
        )

        ax.add_collection3d(
            brain
        )


def add_regions(
    ax,
    metric,
):
    """
    Plot all 68 empirical regions at their
    Desikan-Killiany parcel centroids.
    """

    ax.scatter(
        data["x"],
        data["y"],
        data["z"],
        c=data[metric],
        cmap=cmap,
        norm=norm,
        s=NODE_SIZE,
        edgecolors="none",
        linewidths=0,
        depthshade=False,
    )


def configure_camera(
    ax,
    view,
):

    settings = VIEWS[
        view
    ]

    ax.view_init(
        elev=settings["elev"],
        azim=settings["azim"],
    )

    # Fixed coordinates ensure that all maps
    # use identical spatial framing.
    ax.set_xlim(
        -82,
        82,
    )

    ax.set_ylim(
        -112,
        82,
    )

    ax.set_zlim(
        -65,
        85,
    )

    ax.set_box_aspect(
        (
            164,
            194,
            150,
        )
    )

    ax.set_axis_off()


def render_combined(
    metric,
    view,
    output_file,
):
    """Render cortical surface and regional values together."""

    fig = plt.figure(
        figsize=(6, 6),
        facecolor="white",
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    add_brain(
        ax
    )

    add_regions(
        ax,
        metric,
    )

    configure_camera(
        ax,
        view,
    )

    plt.subplots_adjust(
        left=0,
        right=1,
        bottom=0,
        top=1,
    )

    fig.savefig(
        output_file,
        dpi=400,
        bbox_inches="tight",
        pad_inches=0,
        transparent=False,
    )

    plt.close(
        fig
    )


# ============================================================
# EXPORT CORTICAL MAPS
# ============================================================

metrics = {
    "initial": (
        "vulnerability"
    ),
    "residual": (
        "residual_vulnerability"
    ),
}

for metric_name, metric in metrics.items():

    for view in VIEWS:

        output_file = (
            OUTPUT
            / (
                f"combined_"
                f"{metric_name}_"
                f"{view}.png"
            )
        )

        print(
            "Rendering:",
            output_file.name,
        )

        render_combined(
            metric,
            view,
            output_file,
        )


# ============================================================
# STANDALONE COLOUR BAR
# ============================================================

fig, ax = plt.subplots(
    figsize=(1.4, 5)
)

fig.subplots_adjust(
    left=0.45,
    right=0.70,
    bottom=0.05,
    top=0.95,
)

scalar = cm.ScalarMappable(
    norm=norm,
    cmap=cmap,
)

scalar.set_array([])

cb = fig.colorbar(
    scalar,
    cax=ax,
)

cb.set_label(
    "Network-wide CMI disruption\n"
    "(normalized error)",
    fontsize=10,
)

cb.ax.tick_params(
    labelsize=9
)

fig.savefig(
    OUTPUT
    / "shared_colourbar.png",
    dpi=400,
    bbox_inches="tight",
)

fig.savefig(
    OUTPUT
    / "shared_colourbar.pdf",
    bbox_inches="tight",
)

plt.close(
    fig
)


print()
print("DONE")
print("Output directory:")
print(OUTPUT)

print()
print(
    "Exports:\n"
    "  initial vulnerability maps = PNG\n"
    "  residual vulnerability maps = PNG\n"
    "  shared colour bar = PNG and PDF"
)
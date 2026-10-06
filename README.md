# Preservation of Information under Structural Changes in the Human Connectome

Code accompanying the manuscript:

**Preservation of Information under Structural Changes in the Human
Connectome**\
Erik D. Fagerholm, Taro Tezuka, and Milan Brázdil

## Overview

This repository contains the code used to construct the empirical
cortical connectome, define the structural-change and
Ornstein--Uhlenbeck models, analyse distributed and focal structural
perturbations, perform robustness analyses, and generate the
focal-perturbation brain figure.

The analyses use a 68-region cortical structural connectome derived from
Human Connectome Project tractography data distributed by BrainGraph.

## Requirements

Python 3.10 or later is recommended. Install the required Python
packages with:

``` bash
pip install -r requirements.txt
```

## Data

Download the BrainGraph repeated-measures dataset used in the analysis:

https://braingraph.org/static/repeated_10_scale_33.7z

Extract the archive to a local directory. The first script accepts
either a directory containing the `*_repeated10_scale33.graphml` files
or a compatible zip archive containing the GraphML files.

The raw data are not included in this repository.

## Analysis workflow

Run the scripts from the repository root in the following order.

### 1. Build the group connectome

``` bash
python 01_build_group_connectome.py /path/to/graphml_directory
```

This creates the 68-region group structural-connectivity matrix and
group-average anatomical coordinates in `data/derived/`.

### 2. Build the structural-change and dynamical model

``` bash
python 02_build_model.py
```

This constructs the primary structural-change pattern, initializes the
Ornstein--Uhlenbeck model, calculates baseline conditional mutual
information (CMI), and writes the generated model files to
`data/generated/`.

### 3. Distributed structural-change analysis

``` bash
python 03_main_analysis.py
```

This evaluates the primary distributed structural-change analysis for
`N = 5, 10, 20, 40` structural increments.

The associated robustness and control analyses are:

``` bash
python 03b_structural_magnitude_robustness.py
python 03c_fixed_total_adaptive_capacity.py
python 04_structural_pattern_robustness.py
```

These test robustness to structural-change magnitude, hold total
adaptive capacity fixed across values of `N`, and examine alternative
structural-change patterns, respectively.

### 4. Focal perturbation analysis

Run:

``` bash
python 06a_calibrate_targeted_lesions.py
python 06b_definitive_targeted_lesions.py
python 06c_proportional_lesion_control.py
python 07_focal_initialization_robustness.py
python 08_focal_adaptation_robustness.py
```

These scripts calibrate equal-magnitude focal perturbations, perform the
primary 68-region focal analysis, test proportional lesion
normalization, assess robustness of initial-vulnerability rankings
across random parameter initializations, and assess robustness after
dynamical adaptation.

## Figure generation

`make_focal_brain_figure.py` generates the cortical visualization of
initial and residual vulnerability. It requires a local FreeSurfer
`fsaverage` surface containing the required `surf/` and `label/` files.
These FreeSurfer files are not distributed in this repository.

## Reproducibility

Random number generation used to initialize regional gain and
stochastic-input amplitudes is explicitly seeded where required. The
primary group-connectome construction uses seed `20260907`.

`verify_reconstruction.py` provides additional checks of the
reconstructed analysis inputs and outputs.

## Output directories

Generated data and results are written beneath:

``` text
data/derived/
data/generated/
results/
```

These generated outputs are excluded from version control because they
can be reconstructed from the source data and scripts.

## Code availability

The analysis code is available at:

https://github.com/allavailablepubliccode/Structure_Dynamics_2

# Demo of the Multiscale Multi-species Spatial Signatures (M<sup>2</sup>S<sup>2</sup>) Pipeline

This repository is meant to accompany the paper ["Topology of Multi-species Localization"](http://arxiv.org/abs/2603.032).

## Getting started

The project requires Python 3.13.

The recommended way to set up the project environment is through [uv](https://github.com/astral-sh/uv), which is a tool that can create and manipulate Python virtual environments, manage Python versions, and install Python tools system-wide.

<details>
<summary>Linux/MacOS</summary>

```bash
# Install uv using the instructions at https://docs.astral.sh/uv/getting-started/installation/.
wget -qO- https://astral.sh/uv/install.sh | sh

# Clone the repository.
# IMPORTANT: Replace /project/directory/on/your/computer with the actual path where you want to clone the project.
project_dir="/path/to/project/directory/on/your/computer"
git clone https://github.com/abhinavnatarajan/adenoma_carcinoma "$project_dir"

# Navigate to the project directory
cd "$project_dir" || exit 1

# Create a virtual environment and install all dependencies
uv sync
```

</details>

<details>
<summary>Windows</summary>

```powershell
# Install uv using the instructions at https://docs.astral.sh/uv/getting-started/installation/.
powershell -ExecutionPolicy ByPass -c "Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression"

# Clone the repository.
# IMPORTANT: Replace C:\project\directory\on\your\computer with the actual path where you want to clone the project.
$projectDir = "C:\project\directory\on\your\computer"
git clone https://github.com/abhinavnatarajan/adenoma_carcinoma $projectDir

# Navigate to the project directory
Set-Location $projectDir

# Create a virtual environment and install all dependencies
uv sync
```

</details>

Run all commands below from the repository root after setting up the environment.
Statistics generation and model fitting are computationally intensive. The examples
use explicit output paths; choose separate directories for different configurations.
Statistics generation and colorectal classification resume existing outputs by
default. Run the download steps when setting up a dataset; they replace existing
copies at their default locations.

## Synthetic Data from Tumor Microenvironment

The ABM pipeline computes persistent statistics for single cell types, pairs, and
triples, then clusters the simulations with KMeans. It uses zero imputation and
feature standardization without PCA. Known EEE annotations are used to evaluate
the clusters and calculate grouped permutation importances using ARI, NMI, and
Binder loss. They are excluded from model fitting.

```bash
# Download the simulation point clouds and parameter table.
uv run python -m abm_dataset_analysis.download_data

# Generate the persistent statistics for all singles, pairs, and triples.
uv run python -m abm_dataset_analysis.generate_stats \
    --min-num-labels 1 \
    --max-num-labels 3 \
    --output-dir abm_dataset_analysis/stats_v2

# Fit three clusters and calculate grouped permutation importances.
uv run python -m abm_dataset_analysis.eee_clustering \
    --stats-dir abm_dataset_analysis/stats_v2 \
    --output-file abm_dataset_analysis/results/eee_prediction_results_v2.json \
    --k 3 \
    --num-permutations 100
```

Statistics generation uses the included
[EEE annotation CSV](abm_dataset_analysis/data/ground_truth_labels.csv) by default.
Use `--eee-labels-file` to supply another CSV with `chi_c^m`, `c_half`, and `label`
columns.

Clustering writes a JSON file containing cluster assignments, partition scores,
parameter-group summaries, and permutation importances for each
`(cell group, diagram, dimension)` combination. Choose a new output filename for
each run: the clustering script refuses to overwrite an existing file.

Each permutation shuffles one feature group jointly across simulations and refits
a fresh copy of the full KMeans pipeline, including imputation and scaling, with
the same clustering settings and seed. Importances are baseline minus permuted
ARI/NMI, or permuted minus baseline Binder loss; positive values indicate worse
agreement with the EEE annotations after refitting. New outputs identify this
method with `"permutation_importance_method": "refit_kmeans"`. Results from the
earlier fixed-centroid method must be regenerated to use the refitting method.

### Interactive analysis with marimo

Launch either notebook from the repository root after setting up the environment.
Each command opens the marimo editor in your browser; use a separate terminal for
each notebook if you want both open, and press `Ctrl+C` in its terminal to stop
the server.

```bash
# Explore simulation point clouds and compare representative EEE regimes.
uv run marimo edit abm_dataset_analysis/visualize_examples.py

# Explore clustering results and grouped permutation importances.
uv run marimo edit abm_dataset_analysis/visualize_results.py
```

- [`visualize_examples.py`](abm_dataset_analysis/visualize_examples.py) displays
  simulation grids and the representative equilibrium, elimination, and escape
  examples. It reads `data/simulation_params.csv` and `data/simulation_samples/`
  within `abm_dataset_analysis`, produced by the download command above.
- [`visualize_results.py`](abm_dataset_analysis/visualize_results.py) displays
  subjective and predicted regime maps, cluster purity, partition scores, and
  permutation-importance charts with metric and group-count controls. It reads
  `data/simulation_params.csv` and a saved clustering JSON file within
  `abm_dataset_analysis`; it does not rerun clustering.

The results notebook currently selects
`results/eee_clustering_results_26Oct2026.json`. To view the output of the clustering
command above, change `results_path` in its `load_results` cell to:

```python
results_path = mo.notebook_location() / "results" / "eee_prediction_results_v2.json"
```

Use the corresponding filename if you chose another output path. The notebook
identifies whether the saved importances use KMeans refits or the legacy
fixed-centroid method. Both notebooks display interactive Plotly figures inline.

## Colorectal Cancer Dataset

The colorectal pipeline fits a classifier for each patient using grouped PCA,
records stratified cross-validation scores, and calculates permutation importances
for cell groups. The commands below compare features from singles and pairs with
features from singles, pairs, and triples.

```bash
# Download the dataset.
uv run python -m colorectal_cancer_dataset_analysis.download_data

# Generate persistent statistics for all singles and pairs.
uv run python -m colorectal_cancer_dataset_analysis.generate_stats \
    --max-num-labels 2 \
    --output-dir colorectal_cancer_dataset_analysis/stats_singles_and_pairs_v2

# Run the classification using the persistent statistics from singles and pairs.
uv run python -m colorectal_cancer_dataset_analysis.classification_by_patient_pca \
    --stats-dirs colorectal_cancer_dataset_analysis/stats_singles_and_pairs_v2 \
    --output-dir colorectal_cancer_dataset_analysis/results/classification_pairs_v2

# Generate persistent statistics for all triples.
uv run python -m colorectal_cancer_dataset_analysis.generate_stats \
    --min-num-labels 3 \
    --max-num-labels 3 \
    --output-dir colorectal_cancer_dataset_analysis/stats_triples_v2

# Run the classification using the persistent statistics from singles, pairs and triples.
uv run python -m colorectal_cancer_dataset_analysis.classification_by_patient_pca \
    --stats-dirs colorectal_cancer_dataset_analysis/stats_singles_and_pairs_v2 \
        colorectal_cancer_dataset_analysis/stats_triples_v2 \
    --output-dir colorectal_cancer_dataset_analysis/results/classification_triples_v2
```

Persistent statistics and classification results are Parquet datasets stored in
directories. Classification results are partitioned by `patient_id`; each patient
record contains scores and grouped permutation importance distributions.

Open the interactive Marimo notebooks to explore the classification results:

```bash
uv run marimo edit colorectal_cancer_dataset_analysis/visualize_results_pairs.py
uv run marimo edit colorectal_cancer_dataset_analysis/visualize_results_triples.py
```

These notebooks read `results/classification_pairs_v2` and
`results/classification_triples_v2` within `colorectal_cancer_dataset_analysis`,
matching the output paths above. Update their input paths if you use different
result directories, and launch them from the repository root.

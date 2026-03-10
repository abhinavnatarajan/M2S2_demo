# Demo of the Multiscale Multi-species Spatial Signatures Pipeline

This repository is meant to accompany the paper ["Topology of Multi-species Localization"](http://arxiv.org/abs/2603.032).

## Getting started
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

All scripts in the rest of these instructions assume that you have performed these steps, and must be run from within the project directory.

## Synthetic Data from Tumor Microenvironment

The following commands perform the analysis on the synthetic tumor microenvironment dataset, including downloading the data, generating persistent statistics for all singles, pairs, and triples, and running the clustering using those statistics.
```bash
cd abm_dataset_analysis || exit 1

# Download the dataset.
uv run python download_data.py

# Generate the persistent statistics for all singles, pairs, and triples.
uv run python generate_stats.py

# Run the clustering
uv run python eee_clustering.py

```
The notebooks `abm_dataset_analysis/generate_example_figures.ipynb` and `abm_dataset_analysis/plot_results.ipynb` contain code to generate the plots in the paper from the results of this analysis.

## Colorectal Cancer Dataset

The following commands perform the analysis on the colorectal cancer dataset, including downloading the data, generating persistent statistics for all singles, pairs and triples, and running the analysis using those statistics.

```bash
cd colorectal_cancer_dataset || exit 1

# Download the dataset.
uv run python download_data.py

# Generate persistent statistics for all singles and pairs.
uv run python generate_stats.py \
    --max-num-labels 2 \
    --output-dir stats_singles_and_pairs

# Run the classification using the persistent statistics from singles and pairs.
uv run python classification_by_patient_pca.py \
    --stats-dirs stats_singles_and_pairs \
    --output-file results/without_triples.h5

# Generate persistent statistics for all triples.
uv run classification/scripts/generate_stats.py \
    --max-num-labels 3 \
    --min-num-labels 3 \
    --output-dir stats_triples

# Run the classification using the persistent statistics from singles, pairs and triples.
uv run classification/scripts/classification_by_patient_pca.py \
    --stats-dirs stats_singles_and_pairs stats_triples \
    --output-file results/with_triples.h5
```
The notebook `colorectal_cancer_dataset_analysis/plot_results.ipynb` contains code to generate the plots in the paper from the results of these analyses.

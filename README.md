# Demo of the Multiscale Multi-species Spatial Signatures Pipeline

This repository is meant to accompany the paper ["Topology of Multi-species Localization"](http://arxiv.org/abs/2603.032).

WARNING: The repo isn't ready yet, but will be updated in the next few days.

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

## Datasets

## Generating Statistics and Classification Pipeline - Adenoma Dataset

```bash
# Generate persistent statistics for all celltype pairs.
uv run adenoma_carcinoma_pipeline/scripts/generate_stats.py \
    --max-num-labels 2 \
    --output-dir adenoma_carcinoma_pipeline/stats_singles_and_pairs

# Run the classification using the persistent statistics from celltype pairs.
uv run classification/scripts/classification_by_patient_pca.py \
    --stats-dirs adenoma_carcinoma_pipeline/stats_singles_and_pairs \
    --output-file classification/results/pairs.h5

# Run the classification using the persistent statistics from celltype pairs
# and ignoring epithelium.
uv run classification/scripts/classification_by_patient_pca.py \
    --stats-dirs adenoma_carcinoma_pipeline/stats_singles_and_pairs \
    --discard-epithelium \
    --output-file adenoma_carcinoma_pipeline/results/pairs_noepi.h5

# Generate persistent statistics for all celltype triples only.
uv run classification/scripts/generate_stats.py \
    --max-num-labels 3 \
    --min-num-labels 3 \
    --output-dir adenoma_carcinoma_pipeline/stats_triples

# Run the classification using the persistent statistics from celltype pairs
# and triples.
uv run classification/scripts/classification_by_patient_pca.py \
    --stats-dirs adenoma_carcinoma_pipeline/stats_singles_and_pairs adenoma_carcinoma_pipeline/stats_triples \
    --output-file adenoma_carcinoma_pipeline/results/pairs_and_triples.h5

# Run the classification using the persistent statistics from celltype pairs
# and triples, and ignoring epithelium.
uv run classification/scripts/classification_by_patient_pca.py \
    --stats-dirs adenoma_carcinoma_pipeline/stats_singles_and_pairs adenoma_carcinoma_pipeline/stats_triples \
    --discard-epithelium \
    --output-file adenoma_carcinoma_pipeline/results/pairs_and_triples_noepi.h5
```

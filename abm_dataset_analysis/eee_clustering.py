#! /usr/bin/env python
"""Script to predict tumour containment, equilibrium, or escape."""

import argparse
import json
import logging
import math
import pprint
import sys
import time
from os import PathLike
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import svd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

from utils.logging import configure_logger
from utils.read_stats import normalize_colname, read_stats


def discard_feature(dgm: str, dim: int, statistic: str, cell_group: tuple[str, ...]) -> bool:
    """Check if a feature should be discarded."""
    return (
        dgm in ("cod", "rel")  # discard codomain and relative diagrams
        or (
            dim == 0  # dimension 0
            and dgm == "cok"  # cokernel in dimension 0 is always empty
        )
        or (
            dim == 0  # dimension 0
            and dgm in ("dom", "im")
            and statistic.split("_", maxsplit=1)[0] in ("birth", "midpt", "length")
            # births are always 0 in dom0 and im0
            # midpts and length not required since we have death
        )
        or (
            "iqr" in statistic  # drop interquartile range since we have p25 and p75
        )
        or (
            len(cell_group) == 1  # single cell type
            and dgm != "dom"  # keep only the domain
        )
        or (
            len(cell_group) > 1  # more than one cell type
            and dgm == "dom"  # don't need the domain diagrams for pairs or triples
        )
    )


def read_all_data(
    stats_dir: PathLike,
    labels_include: tuple[str, ...] = (),
    labels_exclude: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Read all the data into one table."""
    logger.info("Reading data")
    data: pd.DataFrame = (
        read_stats(stats_dir)
        .read()
        .to_pandas()
        .drop(
            columns=[
                "filtration_algorithm",  # all are delcech
            ],
        )
    )
    # Convert the codomain to a string
    data["codomain"] = pd.Series(
        ["/".join(sorted(cod_types)) for cod_types in data["codomain"]],
        dtype="string",
        index=data.index,  # otherwise will get weird jumps
    )

    # Pivot the table by codomain
    data = data.pivot_table(
        index=["ID", "M2_ratio", "c_half", "chi_c^m", "label"],
        columns=["codomain"],
        observed=True,
    )

    data.columns = [normalize_colname(c) for c in data.columns]
    discard_columns = [c for c in data.columns if discard_feature(*c)]
    if labels_include:
        discard_columns += [c for c in data.columns if not set(c[-1]).issubset(labels_include)]
    if labels_exclude:
        discard_columns += [c for c in data.columns if set(c[-1]).intersection(labels_exclude)]

    return data.drop(columns=discard_columns)


def pca(
    data: pd.DataFrame,
    n_components: int,
) -> pd.DataFrame:
    """Run PCA on the data and return the transformed data and the PCA object."""
    data_array = data.to_numpy()
    data_mean = np.mean(data_array, axis=0)
    data_centred = data - data_mean
    (_, _, v) = svd(data_centred, lapack_driver="gesvd")
    data_pca = data_centred @ v[:, 0:n_components] + data_mean[0:n_components]
    return pd.DataFrame(data_pca)


def classification_preprocess(
    data: pd.DataFrame,
    pca_size_ratio: float,
    min_features_retain: int,
    imputer_params: dict,
) -> pd.DataFrame:
    """Preprocess the data from a single patient for classification."""
    # Impute missing values
    imputer = SimpleImputer(**imputer_params)
    data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    # Compute the PCA per feature group
    cell_groups: list[tuple[str, ...]] = list({c[-1] for c in data.columns})

    def n_components(n: int) -> int:
        """Return the number of components to keep for PCA."""
        return max(math.ceil(n * pca_size_ratio), min(n, min_features_retain))

    logger = logging.getLogger(__name__)
    data_new = pd.DataFrame()
    for cell_group in cell_groups:
        logger.debug("Running PCA for cell group %s", cell_group)
        cell_group_columns = [c for c in data.columns if c[-1] == cell_group]
        data_cell_group = data.loc[:, cell_group_columns]
        cell_group_pca = pca(data_cell_group, n_components(data_cell_group.shape[1]))
        data = data.drop(columns=cell_group_columns)
        new_colnames = [(cell_group, "pca_" + str(i)) for i in range(cell_group_pca.shape[1])]
        data_cell_group_new = pd.DataFrame(cell_group_pca)
        data_cell_group_new.columns = new_colnames
        data_new = pd.concat([data_new, data_cell_group_new], axis=1)

    return pd.concat([data, data_new], axis=1)


def run_clustering(
    *,
    stats_dir: PathLike,
    labels_include: tuple[str, ...],
    labels_exclude: tuple[str, ...],
    k: int,
    use_pca: bool,
    pca_size_ratio: float,
    min_features_retain: int,
) -> dict:
    """Run classification for each patient."""
    logger = logging.getLogger(__name__)
    clustering_params = {
        "n_clusters": k,
        "max_iter": 100000,
        "algorithm": "elkan",
    }
    imputer_params = {
        "strategy": "constant",
        "fill_value": 0.0,
        "keep_empty_features": True,
    }
    all_params = {
        "ClusteringParams": clustering_params,
        "SimpleImputer": imputer_params,
    }
    data = read_all_data(
        stats_dir=stats_dir,
        labels_include=labels_include,
        labels_exclude=labels_exclude,
    )

    logger.info(
        "Using the following classifier parameters:\n%s",
        pprint.pformat(all_params),
    )
    clustering_fn = KMeans(
        **clustering_params,
    )

    # Preprocess the data
    if use_pca:
        data_raw = classification_preprocess(
            data,
            pca_size_ratio,
            min_features_retain,
            imputer_params,
        )
    else:
        data_raw = pd.DataFrame(
            SimpleImputer(**imputer_params).fit_transform(data),
            columns=data.columns,
        )

    logger.info("Starting clustering.")
    # Train a classifier on the data and cross-validate
    clustering_fn.fit(data_raw)
    if clustering_fn.labels_ is None or clustering_fn.labels_.size == 0:
        logger.error("Clustering failed to produce labels.")
        sys.exit(1)
    predicted_labels = [label.item() for label in clustering_fn.labels_]

    # Group by the parameter combinations and analyze clustering results
    param_combinations = data.index.droplevel(["ID", "label", "M2_ratio"]).unique()
    param_names = param_combinations.names

    results = []

    for param0, param1 in param_combinations:
        # Get all samples with this parameter combination
        param_mask = (data.index.get_level_values(param_names[0]) == param0) & (
            data.index.get_level_values(param_names[1]) == param1
        )
        param_samples = data[param_mask]

        # Get the expected ground truth label (should be constant for this combination)
        ground_truth_labels = param_samples.index.get_level_values("label").unique()
        if len(ground_truth_labels) != 1:
            errmsg = (
                "Multiple ground truth labels for "
                f"{param_names[0]}={param0}, {param_names[1]}={param1}: {ground_truth_labels}"
            )
            logger.warning(errmsg)
        expected_label = ground_truth_labels[0]

        # Get the predicted labels for this parameter combination
        predicted_labels_for_params = [
            predicted_labels[data.index.get_loc(idx)] for idx in param_samples.index
        ]

        # Find the most common predicted label
        unique_labels, counts = np.unique(predicted_labels_for_params, return_counts=True)
        most_common_idx = np.argmax(counts)
        predicted_class = unique_labels[most_common_idx].item()

        # Calculate purity (ratio of most common class to total samples)
        purity = (counts[most_common_idx] / len(predicted_labels_for_params)).item()

        # Store results
        results.append(
            {
                param_names[0]: param0,
                param_names[1]: param1,
                "label": expected_label,
                "predicted_class": predicted_class,
                "purity": purity,
            },
        )
    return {"labels": predicted_labels, "results": results}


def _init_logging(args: argparse.Namespace) -> logging.Logger:
    # configure the root logger
    args.logfile_path = Path(args.logfile_dir).resolve()
    Path.mkdir(args.logfile_path, parents=True, exist_ok=True)
    logfile = args.logfile_path.joinpath(
        "eec_classification_{}.log".format(
            time.strftime("%H:%M:%S-%d%b%Y", time.gmtime()),
        ),
    )
    logger = configure_logger(
        logging.getLogger(__name__),
        filename=logfile,
        file_level=logging.DEBUG,
        console_level=args.verbosity,
    )
    logger.info("Logs will be written to %s", logfile)
    return logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    base_path = Path(__file__).parent

    # global settings go here
    def valid_directory(path: str) -> Path:
        """Check if the directory exists."""
        p = Path(path)
        if not p.exists:
            msg = f"Path does not exist: {path}"
            raise argparse.ArgumentTypeError(msg)
        return p

    default_options = {
        "--stats-dir": {
            "default": str(base_path.joinpath("stats")),
            "type": valid_directory,
            "nargs": "?",
            "help": "List of absolute or relative paths to the cell group persistent statistics.",
        },
        "--output-file": {
            "type": str,
            "nargs": "?",
            "default": str(base_path.joinpath("results/eee_prediction_results.json")),
            "help": "Absolute or relative path to the output file to save.",
        },
        "--labels-include": {
            "type": str,
            "nargs": "*",
            "choices": [
                "M1_Macrophage",
                "M2_Macrophage",
                "Macrophage",
                "Tumour",
                "Vessel",
                "Necrotic",
                "Stroma",
            ],
            "help": "List of cell types to include in the regression feature set.",
        },
        "--labels-exclude": {
            "type": str,
            "nargs": "*",
            "choices": [
                "M1_Macrophage",
                "M2_Macrophage",
                "Macrophage",
                "Tumour",
                "Vessel",
                "Necrotic",
                "Stroma",
            ],
            "help": "List of cell types to exclude from the regression feature set.",
        },
        "--k": {
            "type": int,
            "nargs": "?",
            "default": 3,
            "help": "Number of clusters for k-means clustering",
        },
        "--use-pca": {
            "action": "store_true",
            "default": False,
            "help": "Whether to use PCA for dimensionality reduction on the features "
            "from each group of cell types before clustering.",
        },
        "--pca-size-ratio": {
            "type": float,
            "nargs": "?",
            "default": "0.05",
            "help": "Percentage of pca feature vectors to retain.",
        },
        "--min-features-retain": {
            "type": int,
            "nargs": "?",
            "default": 5,
            "help": "Minimum number of pca features to retain regardless of pca_size_ratio.",
        },
        "--logfile-dir": {
            "default": str(base_path.joinpath("logs")),
            "type": str,
            "help": "Directory where log files will be saved.",
        },
        "--verbosity": {
            "default": "INFO",
            "type": str,
            "nargs": "?",
            "choices": ("DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"),
            "help": "Set the verbosity level of the logger. "
            "Only messages of this level and higher will be logged.",
        },
    }

    for k, v in default_options.items():
        parser.add_argument(k, **v)
    args = parser.parse_args()
    logger = _init_logging(args)
    logger.debug("Provided arguments:\n%s", pprint.pformat(vars(args)))
    labels_include = args.labels_include or ()
    labels_exclude = args.labels_exclude or ()
    clustering_results = run_clustering(
        stats_dir=args.stats_dir,
        labels_include=labels_include,
        labels_exclude=labels_exclude,
        k=args.k,
        use_pca=args.use_pca,
        pca_size_ratio=args.pca_size_ratio,
        min_features_retain=args.min_features_retain,
    )
    logger.info(
        "Saving clustering results to %s",
        args.output_file,
    )
    Path(args.output_file).resolve().parent.mkdir(parents=True, exist_ok=True)

    with Path(args.output_file).open("w") as f:
        json.dump(clustering_results, f, indent=4)

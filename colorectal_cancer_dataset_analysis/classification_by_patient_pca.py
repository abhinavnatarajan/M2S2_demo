#! /usr/bin/env python
"""Script to run classification per patient using trichromatic and pair stats reduced with PCA."""

import argparse
import logging
import math
import os
import pprint
import time
from os import PathLike
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import svd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate
from tqdm.auto import tqdm

from utils.logging import configure_logger
from utils.read_stats import discard_feature, normalize_colname, read_stats

plt.style.use("seaborn-v0_8-darkgrid")


def get_patient_ids(
    stat_dump_dir: PathLike,
) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
    """Retrieve the list of patient IDs from the saved stats."""
    logging.getLogger(__name__).info("Reading list of patient IDs.")
    # Read the dataset
    stat_dump_dir = Path(stat_dump_dir)
    return (
        read_stats(stat_dump_dir)
        .read(columns=["patient_id"])
        .to_pandas()
        .drop_duplicates()["patient_id"]
        .to_numpy()
        .astype(int)
    )


# Read the patient tables
def read_patient_data(
    *,
    stat_dump_dir: PathLike,
    patient_id: int,
    labels_include: tuple[str, ...] = (),
    labels_exclude: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Read the data for a given patient."""
    logging.getLogger().debug("Reading data for patient id %s from %s.", patient_id, stat_dump_dir)
    patient_data: pd.DataFrame = (
        read_stats(stat_dump_dir, filters=[("patient_id", "=", patient_id)])
        .read()
        .to_pandas()
        .drop(
            columns=[
                "filtration_algorithm",  # all are delcech
                "patient_id",  # entire table is only one patient
            ],
        )
        # Drop rows that have empty codomain
        # These correspond to samples with at most one cell type having more than 3 cells
        .dropna(subset=["codomain"])
    )
    # Convert the codomain to a string
    patient_data["codomain"] = pd.Series(
        ["/".join(sorted(cod_types)) for cod_types in patient_data["codomain"]],
        dtype="string",
        index=patient_data.index,  # otherwise will get weird jumps
    )

    # Pivot the table by codomain
    patient_data = patient_data.pivot_table(
        index=["sample_id", "sample_type"],
        columns=["codomain"],
        observed=False,
    )

    patient_data.columns = [normalize_colname(c) for c in patient_data.columns]
    discard_columns = [c for c in patient_data.columns if discard_feature(*c)]
    if labels_include:
        discard_columns += [
            c for c in patient_data.columns if not set(c[-1]).issubset(labels_include)
        ]
    if labels_exclude:
        discard_columns += [
            c for c in patient_data.columns if set(c[-1]).intersection(labels_exclude)
        ]
    return patient_data.drop(columns=discard_columns)


def read_patient_data_combined(
    stat_dump_dirs: list[PathLike],
    labels_include: tuple[str, ...],
    labels_exclude: tuple[str, ...],
    patient_id: int,
) -> pd.DataFrame:
    """Read both pair data and trichromatic data for a given patient."""
    patient_data_list = [
        read_patient_data(
            stat_dump_dir=p,
            patient_id=patient_id,
            labels_include=labels_include,
            labels_exclude=labels_exclude,
        )
        for p in stat_dump_dirs
    ]
    return pd.concat(patient_data_list, axis=1)


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
    *,
    imputer_params: dict[str, Any],
    keep_epithelium: bool,
) -> tuple[pd.DataFrame, float]:
    """Preprocess the data from a single patient for classification.

    First epithelium data is dropped unless keep_epithelium is true.
    Missing data is imputed by a constant value.
    Then data is grouped by cell tuple, and PCA is run on each group,
    keeping 10% of the features.
    The data is replaced by the PCA projection.

    """
    imputer = SimpleImputer(**imputer_params)

    # Drop epithelium columns if they are not required
    if not keep_epithelium:
        data = data.drop(
            columns=[
                column
                for column in data.columns
                if "Epithelium (imm)" in column[-1] or "Epithelium (str)" in column[-1]
            ],
        )

    # Average percentage of features that are not NaN
    avg_perc_features = np.mean(
        data.agg(
            lambda v: np.count_nonzero(~np.isnan(v)) / len(v),
            axis="columns",
        ).to_numpy(),
        dtype="float",
    )

    # Impute missing values
    data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    # Compute the PCA per feature group
    cell_groups: list[tuple[str, ...]] = list({c[-1] for c in data.columns})

    def n_components(n: int) -> int:
        """Return the number of components to keep for PCA."""
        return max(math.ceil(n * 0.05), min(n, 5))

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

    data = pd.concat([data, data_new], axis=1)
    return data, avg_perc_features


def run_classification_single_patient(
    data: pd.DataFrame,
    *,
    imputer_params: dict[str, Any],
    gradient_boosting_classifier_params: dict[str, Any],
    bagging_classifier_params: dict[str, Any],
    splitter_params: dict[str, Any],
    keep_epithelium: bool,
    n_jobs_crossval: int,
) -> dict[str, Any]:
    """Run classification for a single patient."""
    # Check if the target has only samples of a given type
    gbc = GradientBoostingClassifier(
        **gradient_boosting_classifier_params,
    )
    estimator = BaggingClassifier(gbc, **bagging_classifier_params)
    splitter = StratifiedShuffleSplit(**splitter_params)
    baseline_estimator = DummyClassifier()

    # Prediction target
    y = data.index.get_level_values("sample_type").to_numpy()

    # If the target has only samples of a given type, return an empty dictionary
    if len(np.unique(y)) == 1:
        return {}

    # Compute a baseline score
    baseline = baseline_estimator.fit(data, y).score(data, y)

    # Preprcess the data
    data, avg_perc_features = classification_preprocess(
        data,
        imputer_params=imputer_params,
        keep_epithelium=keep_epithelium,
    )

    # Train a classifier on the data and cross-validate
    cv_scores = cross_validate(
        estimator,
        data,
        y,
        scoring=["accuracy", "balanced_accuracy"],
        cv=splitter,
        n_jobs=n_jobs_crossval,
    )
    cv_accuracy = np.mean(cv_scores["test_accuracy"])
    cv_balanced_accuracy = np.mean(cv_scores["test_balanced_accuracy"])
    estimator = estimator.fit(data, y)
    mdi_vec = np.mean(
        np.stack([e.feature_importances_ for e in estimator.estimators_], axis=0),
        axis=0,
    )
    return {
        "baseline": baseline,
        "cv_mean_accuracy": cv_accuracy.item(),
        "cv_mean_balanced_accuracy": cv_balanced_accuracy.item(),
        "num_samples": data.shape[0],
        "avg_perc_features": avg_perc_features,
    } | dict(zip(data.columns, mdi_vec.tolist(), strict=True))


def run_classification_all_patients(
    *,
    stats_dirs: list[PathLike],
    labels_include: tuple[str, ...],
    labels_exclude: tuple[str, ...],
    keep_epithelium: bool,
    num_workers: int,
) -> pd.DataFrame:
    """Run classification for each patient."""
    logger = logging.getLogger(__name__)
    logger.info("Starting classification.")
    n_crossval_splits = 10
    n_jobs_crossval = min(n_crossval_splits, num_workers)
    gradient_boosting_classifier_params = {
        "loss": "log_loss",
        "n_estimators": 25,
        "learning_rate": 0.4,
        "max_features": 0.03,
        "max_depth": 3,
        "max_leaf_nodes": 6,
        "min_samples_leaf": 5,
    }
    bagging_classifier_params = {
        "n_estimators": 500,
        "n_jobs": max(1, int(num_workers / n_jobs_crossval)),
        "max_samples": 1.0,
        "bootstrap": False,
        "random_state": 0,
    }
    splitter_params = {
        "n_splits": n_crossval_splits,
        "test_size": 0.3,
        "random_state": 0,
    }
    imputer_params = {
        "strategy": "constant",
        "fill_value": 0.0,
        "keep_empty_features": True,
    }
    all_params = {
        "GradientBoostingClassifier": gradient_boosting_classifier_params,
        "BaggingClassifier": bagging_classifier_params,
        "StratifiedShuffleSplit": splitter_params,
        "SimpleImputer": imputer_params,
    }
    logger.info(
        "Using the following classifier parameters:\n%s",
        pprint.pformat(all_params),
    )

    classification_results_list = []
    patient_ids = get_patient_ids(stats_dirs[0])
    for patient_id in tqdm(patient_ids):
        logger.debug(
            "Running classification for patient_id: %s",
            patient_id,
        )
        data = read_patient_data_combined(
            stats_dirs,
            labels_include,
            labels_exclude,
            patient_id.item(),
        )

        result = run_classification_single_patient(
            data,
            imputer_params=imputer_params,
            gradient_boosting_classifier_params=gradient_boosting_classifier_params,
            bagging_classifier_params=bagging_classifier_params,
            splitter_params=splitter_params,
            keep_epithelium=keep_epithelium,
            n_jobs_crossval=n_jobs_crossval,
        )
        if len(result) > 0:
            classification_results_list.append(
                {
                    "patient_id": patient_id.item(),
                }
                | result,
            )
    return pd.DataFrame.from_records(classification_results_list).set_index(
        keys="patient_id",
    )


def _init_logging(args: argparse.Namespace) -> logging.Logger:
    # configure the root logger
    args.logfile_path = Path(args.logfile_dir).resolve()
    Path.mkdir(args.logfile_path, parents=True, exist_ok=True)
    logfile = args.logfile_path.joinpath(
        "{}_{}.log".format(
            Path(__file__).with_suffix("").name,
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

    # global settings go here
    def valid_directory(path: str) -> Path:
        """Check if the directory exists."""
        p = Path(path)
        if not p.exists:
            msg = f"Path does not exist: {path}"
            raise argparse.ArgumentTypeError(msg)
        return p

    base_path = Path(__file__).parent
    default_options = {
        "--stats-dirs": {
            "type": valid_directory,
            "nargs": "+",
            "default": [str(base_path.joinpath("stats"))],
            "help": "List of absolute or relative paths to the cell-group statistics.",
        },
        "--output-file": {
            "type": str,
            "default": str(base_path.joinpath("results", "classification_results.h5")),
            "help": "Absolute or relative path to the output file to save.",
        },
        "--labels-include": {
            "type": str,
            "nargs": "+",
            "choices": [
                "Neutrophil",
                "Macrophage",
                "Cytotoxic T Cell",
                "T Helper Cell",
                "Treg Cell",
                "Epithelium (imm)",
                "Periostin",
                "CD146",
                "CD34",
                "SMA",
                "Podoplanin",
                "Epithelium (str)",
            ],
            "help": "List of cell types to include in the classification feature set.",
        },
        "--labels-exclude": {
            "type": str,
            "nargs": "+",
            "choices": [
                "Neutrophil",
                "Macrophage",
                "Cytotoxic T Cell",
                "T Helper Cell",
                "Treg Cell",
                "Epithelium (imm)",
                "Periostin",
                "CD146",
                "CD34",
                "SMA",
                "Podoplanin",
                "Epithelium (str)",
            ],
            "help": "List of cell types to exclude from the classification feature set.",
        },
        "--keep-epithelium": {
            "action": "store_true",
            "help": "Whether to use epithelium cells in the classification.",
        },
        "--num-workers": {
            "default": 0,
            "type": int,
            "help": "Number of CPUs to use for parallel processing. "
            "Set to 0 to use all available CPUs.",
        },
        "--logfile-dir": {
            "default": str(base_path.joinpath("logs")),
            "type": str,
            "help": "Directory where log files will be saved.",
        },
        "--verbosity": {
            "default": "INFO",
            "type": str,
            "choices": ("DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"),
            "help": "Set the verbosity level of the logger. "
            "Only messages of this level and higher will be logged.",
        },
    }

    for k, v in default_options.items():
        parser.add_argument(k, **v)
    args = parser.parse_args()
    if args.num_workers == 0:
        args.num_workers = len(os.sched_getaffinity(0))
    logger = _init_logging(args)
    logger.debug("Provided arguments:\n%s", pprint.pformat(vars(args)))
    labels_include = args.labels_include or ()
    labels_exclude = args.labels_exclude or ()
    classification_results = run_classification_all_patients(
        stats_dirs=args.stats_dirs,
        labels_include=labels_include,
        labels_exclude=labels_exclude,
        keep_epithelium=args.keep_epithlium,
        num_workers=args.num_workers,
    )
    logger.info(
        "Saving classification results to %s",
        args.output_file,
    )
    Path(args.output_file).resolve().parent.mkdir(parents=True, exist_ok=True)
    classification_results.to_hdf(
        args.output_file,
        key="classification_results_by_patient",
        mode="w",
        complevel=9,
    )

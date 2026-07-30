#! /usr/bin/env python
"""Script to run classification per patient using trichromatic and pair stats reduced with PCA."""

import argparse
import logging
import math
import os
import pprint
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from os import PathLike
from pathlib import Path
from typing import Any, Self

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

# from scipy.linalg import svd
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, make_scorer
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from utils.logging import configure_logger
from utils.read_stats import discard_feature, normalize_colname, read_stats

plt.style.use("seaborn-v0_8-darkgrid")


class GroupwisePCA:
    """Run PCA on the features from each cell group separately."""

    def __init__(self) -> None:
        """Initialize the estimator."""
        self.models = {}

    def fit(self, data: pd.DataFrame) -> Self:
        """Compute the PCA projection operators of each cell group from input data."""
        cell_groups = sorted({column[3] for column in data.columns})

        for cell_group in cell_groups:
            columns = [column for column in data.columns if column[3] == cell_group]

            n_features = len(columns)

            # At most 5% of features, at least min(5, n_features) features
            n_components = max(
                math.ceil(0.05 * n_features),
                min(n_features, 5),
            )
            # Respect the training-sample rank.
            n_components = min(
                n_components,
                n_features,
                max(1, len(data) - 1),
            )

            pipeline = Pipeline(
                [
                    (
                        "standardization",
                        StandardScaler(),
                    ),
                    (
                        "pca",
                        PCA(n_components=n_components),
                    ),
                ],
            )

            pipeline.fit(data.loc[:, columns])

            self.models[cell_group] = {
                "columns": columns,
                "pipeline": pipeline,
                "n_components": n_components,
            }

        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Project each cell group in the input data onto the computed principal components."""
        transformed_groups = []

        for cell_group, model_info in self.models.items():
            columns = model_info["columns"]
            pipeline = model_info["pipeline"]

            values = pipeline.transform(data.loc[:, columns])

            transformed_columns = pd.MultiIndex.from_tuples(
                [(cell_group, f"pca_{component}") for component in range(values.shape[1])],
                names=["cell_group", "component"],
            )

            transformed_groups.append(
                pd.DataFrame(
                    values,
                    index=data.index,
                    columns=transformed_columns,
                ),
            )

        return pd.concat(transformed_groups, axis=1)

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Project each cell group in the input data onto the corresponding principal components."""
        return self.fit(data).transform(data)


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
    # Output features are indexed by (diagram, homological dim, statistic, cell_group)
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


def classification_preprocess(
    data: pd.DataFrame,
    *,
    keep_epithelium: bool,
) -> tuple[pd.DataFrame, float]:
    """Preprocess the data from a single patient for classification.

    First epithelium data is dropped unless keep_epithelium is true.
    Missing data is imputed by a constant value.
    Then data is grouped by cell tuple, and PCA is run on each group,
    keeping 10% of the features.
    The data is replaced by the PCA projection.

    """
    imputer = SimpleImputer(
        strategy="constant",
        fill_value=0,
        keep_empty_features=True,
    )

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

    return data, avg_perc_features


def run_classification_single_patient(
    data: pd.DataFrame,
    *,
    classifier: BaseEstimator,
    splitter: StratifiedKFold,
    repetitions: int,
    permutations: int,
    keep_epithelium: bool,
) -> dict[str, Any]:
    """Run classification for a single patient."""
    # Check if the target has only samples of a given type
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
        keep_epithelium=keep_epithelium,
    )

    cell_groups = np.unique([c[-1] for c in data.columns])

    # Compute the PCA per feature group
    estimator = Pipeline([("pca", GroupwisePCA()), ("classification", classifier)])

    # Train a classifier on the data and cross-validate
    # Pre-compute the row permutations
    rng = np.random.default_rng(0)
    perms = np.array([rng.permutation(data.shape[0]) for j in range(permutations)])
    cv_scores = np.empty(repetitions, dtype=float)
    group_importance_distributions = dict.fromkeys(cell_groups, np.empty(repetitions, dtype=float))

    def score_func(target: np.ndarray, pred: np.ndarray) -> float:
        return 1.0 - brier_score_loss(target, pred)

    scorer = make_scorer(
        score_func,
        greater_is_better=False,
        response_method="predict_proba",
    )

    def compute_feature_importance_from_cv_result(
        cv_score: float,
        cell_group: str,
    ) -> tuple[str, float]:
        group_importance = 0
        group_columns = [c for c in data.columns if c[-1] == cell_group]

        for perm in tqdm(perms, desc="Permutation", keep=False):
            data_permuted = data.copy()
            data_permuted.loc[:, group_columns] = data_permuted[group_columns].to_numpy()[perm]
            oof_predictions = np.empty(len(y), dtype=float)

            for estimator, idx in zip(estimators, test_indices, strict=True):
                data_test = data_permuted.iloc[idx]
                # predict_proba gives a row of [Pr(0), Pr(1)] for each observation
                oof_predictions[idx] = estimator.predict_proba(data_test)[:, 1]

            # Compute the drop in score
            perm_score = score_func(oof_predictions, y)
            group_importance += cv_score - perm_score

        group_importance /= len(perms)
        return cell_group, group_importance

    for rep in tqdm(np.arange(repetitions), desc="Cross-validation repetition", keep=False):
        n_splits = splitter.get_n_splits(X=data, y=y)
        cv_result = cross_validate(
            estimator,
            data,
            y,
            scoring=scorer,
            cv=splitter,
            n_jobs=n_splits,
            return_estimator=True,
            return_indices=True,
        )
        estimators = cv_result["estimator"]
        test_indices = cv_result["indices"]["test"]
        cv_scores[rep] = np.mean(cv_result["test_score"])
        with ThreadPoolExecutor() as executor:
            group_importances_futures = [
                executor.submit(
                    compute_feature_importance_from_cv_result,
                    cv_scores[rep],
                    cell_group,
                )
                for cell_group in cell_groups
            ]
        for future in tqdm(
            as_completed(group_importances_futures),
            desc="Cell group",
            total=len(cell_groups),
            keep=False,
        ):
            cell_group, cell_group_importance = future.result()
            group_importance_distributions[cell_group][rep] = cell_group_importance

    return {
        "baseline": baseline,
        "scores": cv_scores,
        "feature_importances": group_importance_distributions,
        "num_samples": data.shape[0],
        "avg_perc_features": avg_perc_features,
    }


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

    classification_results_list = []
    patient_ids = get_patient_ids(stats_dirs[0])
    for patient_id in tqdm(patient_ids, desc="Patients"):
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

        gbc = GradientBoostingClassifier(
            loss="log_loss",
            n_estimators=25,
            learning_rate=0.4,
            max_features=0.03,
            max_depth=3,
            max_leaf_nodes=6,
            min_samples_leaf=5,
        )
        classifier = BaggingClassifier(
            gbc,
            n_estimators=500,
            n_jobs=max(1, int(num_workers / 10)),
            max_samples=1.0,
            bootstrap=False,
            random_state=0,
        )
        splitter = StratifiedKFold(n_splits=8, random_state=0)
        repetitions = 50
        permutations = 30
        result = run_classification_single_patient(
            data,
            classifier=classifier,
            splitter=splitter,
            repetitions=repetitions,
            permutations=permutations,
            keep_epithelium=keep_epithelium,
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
        if not p.is_dir():
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
        "--crossval-splits": {
            "type": int,
            "default": 5,
            "help": "Number of folds in each repetition of cross-validation.",
        },
        "--crossval-repetitions": {
            "type": int,
            "default": 50,
            "help": "Number of repetitions for cross-validation.",
        },
        "--crossval-permutations": {
            "type": int,
            "default": 30,
            "help": "Number of permutations to use for computing "
            "grouped permutation importance scores. "
            "In each repetition of cross-validation, "
            "each feature group is permuted this many times."
            "An out-of-fold prediction, is obtained after each permutation "
            "and prediction loss is averaged across permutations. "
            "This yields a matrix of importance scores of shape "
            "[feature_group, repetition_index].",
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
        cpu_count = os.cpu_count()
        args.num_workers = cpu_count - 1 if cpu_count is not None else 1
    logger = _init_logging(args)
    logger.debug("Provided arguments:\n%s", pprint.pformat(vars(args)))
    labels_include = args.labels_include or ()
    labels_exclude = args.labels_exclude or ()
    classification_results = run_classification_all_patients(
        stats_dirs=args.stats_dirs,
        labels_include=labels_include,
        labels_exclude=labels_exclude,
        keep_epithelium=args.keep_epithelium,
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

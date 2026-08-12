#! /usr/bin/env python
"""Script to run classification per patient using trichromatic and pair stats reduced with PCA."""

import argparse
import logging
import math
import os
import pprint
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Any, Self

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from utils.logging import configure_logger
from utils.persistent_stats import PERS_STATS_NAMES, STATISTIC_NAMES

plt.style.use("seaborn-v0_8-darkgrid")

__all__ = [
	"get_patient_ids",
	"read_patient_data",
	"run_classification_single_patient",
]


class GroupwisePCA(BaseEstimator):
	"""Run PCA on the features from each cell group separately."""

	def __init__(self) -> None:
		"""Initialize the estimator."""
		self.models = {}

	def fit(self, data: pd.DataFrame) -> Self:
		"""Compute the PCA projection operators of each cell group from input data."""
		cell_groups = sorted({column[-1] for column in data.columns})

		for cell_group in cell_groups:
			columns = [column for column in data.columns if column[-1] == cell_group]

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
	*stat_dump_dirs: PathLike,
) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
	"""Retrieve the list of patient IDs from the saved stats."""
	logging.getLogger(__name__).info("Reading list of patient IDs.")
	# Read the dataset
	return (
		pl.scan_parquet([Path(path) for path in stat_dump_dirs], hive_partitioning=True)
		.select(
			"patient_id",
		)
		.unique()
		.collect()
		.get_column("patient_id")
		.to_numpy()
		.astype(int)
	)


# Read the patient tables
def read_patient_data(
	*,
	stat_dump_dirs: Sequence[PathLike],
	patient_id: int,
	labels_include: tuple[str, ...] = (),
	labels_exclude: tuple[str, ...] = (),
	keep_epithelium: bool = False,
) -> pl.DataFrame:
	"""Read the data for a given patient."""
	# Drop epithelium columns if they are not required
	if not keep_epithelium:
		labels_exclude += ("Epithelium (imm)", "Epithelium (str)")

	# Filters a row based on the codomain cell group
	filter_labels_include = pl.col("codomain").list.set_difference(labels_include).list.len().eq(0)
	filter_labels_exclude = (
		pl.col("codomain").list.set_intersection(labels_exclude).list.len().eq(0)
	)

	# Some columns can be universally dropped without filtering the cell-groups.
	# We can do this pre-pivot to save time.
	dgm_names = r"(ker|cok|dom|cod|im|rel)"
	pers_stats_names = rf"({('|').join(PERS_STATS_NAMES)})"
	statistic_names = rf"({('|').join(STATISTIC_NAMES)})"
	# Codomain and relative diagrams
	cod_dgm = cs.matches(rf"cod\d+-{pers_stats_names}$")
	rel_dgm = cs.matches(rf"rel\d+-{pers_stats_names}$")
	cod_or_rel = cod_dgm | rel_dgm
	# Cokernel in degree 0
	cokernel_dim0 = cs.matches(rf"cok0-{pers_stats_names}$")
	# Births are always 0 in dom0 and im0
	# Midpts and length not required since we have death.
	redundant_dim0_stats = cs.matches(rf"^(dom|im)0-{statistic_names}_(birth|midpt|length)$")
	# Interquartile range not needed since we have p25 and p75
	iqr = cs.matches(rf"^{dgm_names}\d+-iqr$")
	pre_pivot_drop_cols = cod_or_rel | cokernel_dim0 | redundant_dim0_stats | iqr

	# Post-pivot column selectors to drop
	# Drop non-domain diagrams for single cell types
	single_cell_type = cs.matches(rf"^{dgm_names}\d+-{pers_stats_names}-[^/]+$")
	non_dom_single_cell_type = single_cell_type - cs.matches(rf"dom\d+-{pers_stats_names}-.+$")
	# Drop dom diagram for multiple cell types
	dom_multiple_cell_types = cs.matches(rf"^dom\d+-{pers_stats_names}-[^/]+/.+$")
	post_pivot_drop_cols = non_dom_single_cell_type | dom_multiple_cell_types

	patient_dfs: list[pl.LazyFrame] = []
	for path in stat_dump_dirs:
		logging.getLogger().debug("Reading data for patient id %s from %s.", patient_id, path)
		stat_path = Path(path).resolve()
		df: pl.LazyFrame = (
			pl.scan_parquet(
				stat_path,
				hive_partitioning=True,
				missing_columns="insert",
			)
			.filter(pl.col("patient_id") == patient_id)
			.drop("filtration_algorithm", "patient_id", pre_pivot_drop_cols)
			.drop_nulls(subset="codomain")
			.filter(filter_labels_include & filter_labels_exclude)
		)
		patient_dfs.append(df)

	# Output features are indexed by "<diagram><dim>-<statistic>_<variable>-<cell/group/types>"
	return (
		pl.concat(patient_dfs, how="vertical", rechunk=True)  # ruff: ignore[PD010]
		.with_columns(pl.col("codomain").list.sort().list.join("/"))
		.collect()
		.pivot("codomain", index=["sample_type", "sample_id"], separator="-")
		.drop(post_pivot_drop_cols)
	)


def classification_preprocess(
	data: pd.DataFrame,
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


@dataclass(slots=True)
class ClassificationParams:
	"""Parameters for classification."""

	classifier: BaseEstimator = field(kw_only=True)
	"""Base model for classification."""
	splitter: StratifiedKFold = field(kw_only=True)
	"""Splitter for each cross-validation repetition."""
	cross_val_repeats: int = field(kw_only=True)
	"""Number of cross-validation repetitions."""
	num_permutations: int = field(kw_only=True)
	"""Number of random permutations to use to estimate permutation importance scores."""
	score_func: Callable[[np.ndarray, np.ndarray], float] = field(kw_only=True)
	"""Score function to evaluate a prediction."""


def run_classification_single_patient(
	data: pd.DataFrame,
	*,
	classification_params: ClassificationParams,
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
	data, avg_perc_features = classification_preprocess(data)

	cell_groups = np.unique([c[-1] for c in data.columns])

	# Train a classifier on the data and cross-validate
	# Pre-compute the row permutations
	rng = np.random.default_rng(0)
	perms = np.array(
		[rng.permutation(data.shape[0]) for j in range(classification_params.num_permutations)],
	)
	cv_scores = np.empty(classification_params.cross_val_repeats, dtype=float)
	group_importance_distributions = dict.fromkeys(
		cell_groups,
		np.empty(classification_params.cross_val_repeats, dtype=float),
	)

	scorer = make_scorer(
		classification_params.score_func,
		greater_is_better=False,
		response_method="predict_proba",
	)

	def compute_feature_importance_from_cv_result(
		cv_score: float,
		feature_group: str,
	) -> tuple[str, float]:
		score_from_random_permutations = 0
		group_importance = 0
		group_columns = [c for c in data.columns if c[-1] == feature_group]

		# Compute the classification score after permuting the rows of the data
		# within the columns corresponding to this feature group
		for perm in tqdm(perms, desc="Permutation", keep=False):
			data_permuted = data.copy()
			data_permuted.loc[:, group_columns] = data_permuted[group_columns].to_numpy()[perm]
			oof_predictions = np.empty(len(y), dtype=float)

			for estimator, idx in zip(estimators, test_indices, strict=True):
				data_test = data_permuted.iloc[idx]
				# predict_proba gives a row of [Pr(0), Pr(1)] for each observation
				oof_predictions[idx] = estimator.predict_proba(data_test)[:, 1]

			score_from_random_permutations += classification_params.score_func(oof_predictions, y)

		# Group importance is the the average drop in score across permutations of the input
		group_importance = cv_score - score_from_random_permutations / len(perms)
		return feature_group, group_importance

	for rep in tqdm(
		np.arange(classification_params.cross_val_repeats),
		desc="Cross-validation repetition",
		keep=False,
	):
		n_jobs = classification_params.splitter.get_n_splits(X=data, y=y)
		cv_result = cross_validate(
			classification_params.classifier,
			data,
			y,
			scoring=scorer,
			cv=classification_params.splitter,
			n_jobs=n_jobs,
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
	cross_val_repeats: int,
	cross_val_splits: int,
	num_permutations: int,
	num_workers: int,
) -> pd.DataFrame:
	"""Run classification for each patient."""
	logger = logging.getLogger(__name__)
	logger.info("Starting classification.")

	classification_results_list = []
	patient_ids = get_patient_ids(*stats_dirs)
	for patient_id in tqdm(patient_ids, desc="Patients"):
		logger.debug(
			"Running classification for patient_id: %s",
			patient_id,
		)
		data = read_patient_data(
			stat_dump_dirs=stats_dirs,
			labels_include=labels_include,
			labels_exclude=labels_exclude,
			keep_epithelium=keep_epithelium,
			patient_id=patient_id.item(),
		)

		gradient_booster = GradientBoostingClassifier(
			loss="log_loss",
			n_estimators=25,
			learning_rate=0.4,
			max_features=0.03,
			max_depth=3,
			max_leaf_nodes=6,
			min_samples_leaf=5,
		)
		ensemble_classifier = BaggingClassifier(
			gradient_booster,
			n_estimators=500,
			n_jobs=max(1, int(num_workers / 10)),
			max_samples=1.0,
			bootstrap=False,
			random_state=0,
		)
		classifier = Pipeline(
			[("pca", GroupwisePCA()), ("classification", ensemble_classifier)],
		)
		splitter = StratifiedKFold(n_splits=cross_val_splits, random_state=0)

		def score_func(target: np.ndarray, pred: np.ndarray) -> float:
			return 1.0 - brier_score_loss(target, pred)

		classification_params = ClassificationParams(
			classifier=classifier,
			splitter=splitter,
			cross_val_repeats=cross_val_repeats,
			num_permutations=num_permutations,
			score_func=score_func,
		)
		result = run_classification_single_patient(
			data,
			classification_params=classification_params,
		)
		if len(result) > 0:
			classification_results_list.append(
				{
					"patient_id": patient_id.item(),
				}
				| result,
			)
	return pd.json_normalize(classification_results_list).set_index(
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
		"--cross-val-splits": {
			"type": int,
			"default": 8,
			"help": "Number of folds in each repetition of cross-validation.",
		},
		"--cross-val-repeats": {
			"type": int,
			"default": 50,
			"help": "Number of cross-validation repetitions.",
		},
		"--num-permutations": {
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
		num_permutations=args.num_permutations,
		cross_val_repeats=args.cross_val_repeats,
		cross_val_splits=args.cross_val_splits,
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

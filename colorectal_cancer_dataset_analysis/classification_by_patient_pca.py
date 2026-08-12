#! /usr/bin/env python
"""Script to run classification per patient using trichromatic and pair stats reduced with PCA."""

import argparse
import logging
import math
import os
import pprint
import re
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Any, ClassVar, Self, TypeGuard, cast, get_args

import numpy as np
import polars as pl
import polars.selectors as cs
import sklearn
from chalc.sixpack.types import DiagramName
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
from sklearn.metrics import brier_score_loss, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from tqdm.auto import tqdm

from utils.logging import configure_logger
from utils.persistent_stats import PERS_STATS_NAMES, STATISTIC_NAMES

sklearn.set_config(enable_metadata_routing=True)

__all__ = [
	"get_patient_ids",
	"read_patient_data",
	"run_classification_all_patients",
	"run_classification_single_patient",
]

SAMPLE_TYPES_ENUM = pl.Enum(["adenoma", "cancer"])


@dataclass(slots=True)
class FeatureParts:
	"""Lazily parse a feature name into its components."""

	dgm_and_dim: str
	_dgm: DiagramName | None
	_dim: int | None
	statistic: str
	codomain: str
	NUM_FEATURE_NAME_PARTS: ClassVar[int] = 3
	DGM_NAMES_REGEX: ClassVar[str] = "|".join(
		re.escape(name) for name in get_args(DiagramName.__value__)
	)

	@classmethod
	def is_diagram_name(cls, name: str) -> TypeGuard[DiagramName]:
		"""Check if a string is a diagram name."""
		return name in get_args(DiagramName.__value__)

	def __init__(self, feature_name: str) -> None:
		# Expects feature names of the form <dgm><dim>-<statistic>-<cell_type/cell_type/...>
		# such as ker0-avg_birth-cell1/cell2/cell3
		tokens = feature_name.split("-", maxsplit=2)
		if len(tokens) != self.NUM_FEATURE_NAME_PARTS:
			errmsg = f"Invalid feature name: {feature_name!r}"
			raise ValueError(errmsg)
		self.dgm_and_dim = tokens[0]
		self._dgm = None
		self._dim = None
		self.statistic = tokens[1]
		self.codomain = tokens[2]

	def _parse_dgm_and_dim(self) -> None:
		if self._dim is not None and self._dgm is not None:
			return
		match = re.fullmatch(rf"({self.DGM_NAMES_REGEX})(\d+)", self.dgm_and_dim)
		if not match:
			errmsg = f"Invalid diagram and dimension: {self.dgm_and_dim!r}"
			raise ValueError(errmsg)
		self._dgm = cast("DiagramName", match.group(1))
		self._dim = int(match.group(2))

	@property
	def dim(self) -> int:
		self._parse_dgm_and_dim()
		return cast("int", self._dim)

	@property
	def dgm(self) -> DiagramName:
		self._parse_dgm_and_dim()
		return cast("DiagramName", self._dgm)


class GroupwisePCA(BaseEstimator):
	"""Run PCA on the features from each cell group separately."""

	def __init__(self) -> None:
		"""Initialize the estimator."""

	def fit(
		self,
		data: pl.DataFrame,
		y: Any = None,  # noqa: ARG002
		*,
		cell_group_columns: dict[str, Sequence[str]] | None = None,
	) -> Self:
		"""Compute the PCA projection operators of each cell group from input data."""
		if not cell_group_columns:
			errmsg = "GroupwisePCA.fit requires non-empty cell_group_columns metadata."
			raise ValueError(errmsg)

		self.models_ = {}

		for cell_group, group_columns in cell_group_columns.items():
			columns = list(group_columns)
			n_features = len(columns)
			if n_features == 0:
				errmsg = f"Cell group {cell_group!r} has no feature columns."
				raise ValueError(errmsg)

			# Keep at least five components, or 5% for larger feature groups.
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

			pipeline.fit(data.select(columns))

			self.models_[cell_group] = {
				"columns": columns,
				"pipeline": pipeline,
				"n_components": n_components,
			}

		return self

	def transform(self, data: pl.DataFrame) -> pl.DataFrame:
		"""Project each cell group in the input data onto the computed principal components."""
		check_is_fitted(self, "models_")
		transformed_groups = []

		for cell_group, model_info in self.models_.items():
			columns = model_info["columns"]
			pipeline = model_info["pipeline"]

			values = pipeline.transform(data.select(columns))
			new_column_names = [
				f"pca_{component}-{cell_group}" for component in range(values.shape[1])
			]

			transformed_groups.append(
				pl.DataFrame(
					values,
					schema=new_column_names,
				),
			)

		return pl.concat(transformed_groups, how="horizontal").rechunk()

	def fit_transform(
		self,
		data: pl.DataFrame,
		y: Any = None,
		*,
		cell_group_columns: dict[str, Sequence[str]] | None = None,
	) -> pl.DataFrame:
		"""Project each cell group in the input data onto the corresponding principal components."""
		return self.fit(data, y, cell_group_columns=cell_group_columns).transform(data)


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
	codomain_in_labels_include = (
		pl.col("codomain").list.set_difference(labels_include).list.len().eq(0)
		if labels_include
		else pl.lit(value=True)
	)
	codomain_not_in_labels_exclude = (
		pl.col("codomain")
		.list.set_intersection(
			labels_exclude,
		)
		.list.len()
		.eq(0)
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
			.filter(codomain_in_labels_include & codomain_not_in_labels_exclude)
			.cast({"sample_type": SAMPLE_TYPES_ENUM})
		)
		patient_dfs.append(df)

	# Output features are named "<diagram><dim>-<statistic>_<variable>-<cell/group/types>".
	return (
		pl.concat(patient_dfs, how="diagonal", rechunk=True)
		.with_columns(pl.col("codomain").list.sort().list.join("/"))
		.collect()
		.pivot("codomain", index=["sample_type", "sample_id"], separator="-")
		.drop(post_pivot_drop_cols)
	)


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
	score_func: Callable[[Iterable, Iterable], float] = field(kw_only=True)
	"""Score function to evaluate a prediction."""


def run_classification_single_patient(  # noqa: C901, PLR0915
	data: pl.DataFrame,
	*,
	classification_params: ClassificationParams,
) -> dict[str, Any]:
	"""Run classification for a single patient.

	For cross-validation: we repeat stratified K-fold cross-validation
	several times, and record a single cross-validation score from each repetition.
	The score from a single-repetition of cross-validation is computed
	from concatenating the out-of-fold predictions from each fold, and
	comparing with the ground truth via a loss function.

	Feature importance for each cell-group is computed by a grouped permutation
	importance score, averaged across several permutations per cell group.
	In more detail: the importance score for a given cell-group is the drop in
	prediction score of the estimators computed during cross-validation
	(where the score is measured using the concatenated out-of-fold predictions)
	when columns corresponding to the cell-group are permuted along their rows.
	This score is averaged across several permutations per repetition of
	cross-validation, and the distribution of feature importances across
	repetitions is recorded for each cell group.
	"""
	# If data is empty return empty dictionary
	if data.height == 0 or data.width == 0:
		return {}

	# Prediction target
	y = data.get_column("sample_type").to_physical().to_numpy()

	# If the target has only samples of a given type, return an empty dictionary
	classes, class_counts = np.unique(y, return_counts=True)
	if len(classes) == 1:
		return {}
	if len(classes) != len(SAMPLE_TYPES_ENUM.categories):
		errmsg = (
			f"Expected {len(SAMPLE_TYPES_ENUM.categories)} sample types, got {classes.tolist()}."
		)
		raise ValueError(errmsg)
	if classification_params.cross_val_repeats < 1:
		errmsg = "cross_val_repeats must be at least 1."
		raise ValueError(errmsg)
	if classification_params.num_permutations < 1:
		errmsg = "num_permutations must be at least 1."
		raise ValueError(errmsg)
	n_splits = classification_params.splitter.get_n_splits(X=data, y=y)
	if class_counts.min() < n_splits:
		logging.getLogger(__name__).warning(
			"Skipping patient: the smallest class has %d samples, fewer than %d CV folds.",
			class_counts.min(),
			n_splits,
		)
		return {}

	# Get the feature table only and pre-process it
	data = data.drop("sample_type", "sample_id").rechunk()
	if data.width == 0:
		return {}
	avg_perc_features = data.count().to_numpy().mean() / data.height
	data = data.fill_null(strategy="zero")

	# Get a mapping of cell groups to column names
	cell_groups = sorted({FeatureParts(colname).codomain for colname in data.columns})
	cell_group_columns = {cell_group: [] for cell_group in cell_groups}
	for column in data.columns:
		cell_group = FeatureParts(column).codomain
		cell_group_columns[cell_group].append(column)

	# Pre-allocate the list of cross-validation scores and feature importances
	cv_scores = np.empty(classification_params.cross_val_repeats, dtype=float)
	baseline_scores = np.empty(classification_params.cross_val_repeats, dtype=float)
	group_importance_distributions = {
		cell_group: np.full(classification_params.cross_val_repeats, np.nan)
		for cell_group in cell_groups
	}

	scorer = make_scorer(
		classification_params.score_func,
		greater_is_better=True,
		response_method="predict_proba",
	)

	def compute_feature_importance_from_cv_result(
		cv_score: float,
		cell_group: str,
	) -> tuple[str, float]:
		score_from_random_permutations = 0
		group_importance = 0
		group_columns = cell_group_columns[cell_group]

		# Compute the classification score after jointly permuting this group's
		# columns within each held-out fold.
		for permutations_for_folds in fold_permutations:
			oof_predictions = np.zeros(len(y), dtype=float)

			for estimator, test_idxs, permutation in zip(
				estimators,
				test_indices_list,
				permutations_for_folds,
				strict=True,
			):
				data_test = data.gather(test_idxs)
				data_test_permuted = data_test.update(
					data_test.select(group_columns).gather(permutation),
				)
				oof_predictions[test_idxs] = predict_positive_probability(
					estimator,
					data_test_permuted,
				)

			score_from_random_permutations += classification_params.score_func(y, oof_predictions)

		# Group importance is the average drop in score across permutations of the input.
		group_importance = (
			cv_score - score_from_random_permutations / classification_params.num_permutations
		)
		return cell_group, group_importance

	def predict_positive_probability(
		estimator: BaseEstimator,
		data_test: pl.DataFrame,
	) -> np.ndarray:
		classes_ = np.asarray(estimator.classes_)
		positive_class_indices = np.flatnonzero(classes_ == classes.max())
		if len(positive_class_indices) != 1:
			errmsg = f"Could not identify the positive class in {classes_.tolist()}."
			raise RuntimeError(errmsg)
		return estimator.predict_proba(data_test)[:, positive_class_indices.item()]

	for repetition in tqdm(
		np.arange(classification_params.cross_val_repeats),
		desc="Cross-validation repetition",
		leave=False,
	):
		splitter = StratifiedKFold(
			n_splits=n_splits,
			shuffle=True,
			random_state=repetition,
		)
		cv_result = cross_validate(
			classification_params.classifier,
			data,
			y,
			scoring=scorer,
			cv=splitter,
			n_jobs=n_splits,
			return_estimator=True,
			return_indices=True,
			params={"cell_group_columns": cell_group_columns},
		)
		estimators = cv_result["estimator"]
		train_indices_list = cv_result["indices"]["train"]
		test_indices_list = cv_result["indices"]["test"]
		oof_predictions = np.empty(len(y), dtype=float)
		baseline_oof_predictions = np.empty(len(y), dtype=float)
		for estimator, train_idxs, test_idxs in zip(
			estimators,
			train_indices_list,
			test_indices_list,
			strict=True,
		):
			oof_predictions[test_idxs] = predict_positive_probability(
				estimator,
				data.gather(test_idxs),
			)
			baseline_estimator = DummyClassifier().fit(data.gather(train_idxs), y[train_idxs])
			baseline_oof_predictions[test_idxs] = predict_positive_probability(
				baseline_estimator,
				data.gather(test_idxs),
			)
		cv_scores[repetition] = classification_params.score_func(y, oof_predictions)
		baseline_scores[repetition] = classification_params.score_func(y, baseline_oof_predictions)
		fold_permutations = [
			[
				np.random.default_rng(
					(cast("int", repetition), permutation_index, fold_index),
				).permutation(
					len(test_idxs),
				)
				for fold_index, test_idxs in enumerate(test_indices_list)
			]
			for permutation_index in range(classification_params.num_permutations)
		]
		# fold_permutations[i] is the ith permutation for this repetition
		# The permutation is not generic, but only permutes within each fold.
		# fold_permutations[i][j] is the permutation within fold j.

		for cell_group in tqdm(
			cell_groups,
			desc="Cell group",
			leave=False,
		):
			group_name, cell_group_importance = compute_feature_importance_from_cv_result(
				cv_scores[repetition],
				cell_group,
			)
			group_importance_distributions[group_name][repetition] = cell_group_importance

	return {
		"baseline": baseline_scores.mean(),
		"scores": cv_scores,
		"feature_importances": group_importance_distributions,
		"num_samples": data.shape[0],
		"avg_perc_features": avg_perc_features,
	}


def run_classification_all_patients(
	*,
	stats_dirs: list[PathLike],
	labels_include: tuple[str, ...] = (),
	labels_exclude: tuple[str, ...] = (),
	keep_epithelium: bool = False,
	cross_val_repeats: int = 50,
	cross_val_splits: int = 8,
	num_permutations: int = 30,
	num_workers: int = 1,
) -> list[dict]:
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
			n_jobs=max(1, int(num_workers / cross_val_splits)),
			max_samples=1.0,
			bootstrap=False,
			random_state=0,
		)
		classifier = Pipeline(
			[
				(
					"pca",
					GroupwisePCA().set_fit_request(cell_group_columns=True),
				),
				("classification", ensemble_classifier),
			],
		)
		splitter = StratifiedKFold(n_splits=cross_val_splits)

		def score_func(target: Iterable, pred: Iterable) -> float:
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
	return classification_results_list


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
	Path(args.output_file).resolve().parent.mkdir(parents=True, exist_ok=True)
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
	# classification_results.to_hdf(
	# 	args.output_file,
	# 	key="classification_results_by_patient",
	# 	mode="w",
	# 	complevel=9,
	# )

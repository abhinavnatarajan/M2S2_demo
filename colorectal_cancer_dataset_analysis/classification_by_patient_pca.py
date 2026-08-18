#! /usr/bin/env python
"""Script to run classification per patient using trichromatic and pair stats reduced with PCA."""

from __future__ import annotations

import argparse
import logging
import os
import pprint
import re
import time
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Protocol, TypeGuard, cast, get_args

import numpy as np
import polars as pl
import polars.selectors as cs
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn
from chalc.sixpack.types import DiagramName
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm

if TYPE_CHECKING:
	from collections.abc import Callable, Mapping, Sequence

	from numpy.typing import ArrayLike

	type NDArray1[M: np.integer | np.floating] = np.ndarray[tuple[int], np.dtype[M]]

	type ScoreFunc = Callable[[ArrayLike, ArrayLike], float]


from utils.logging import configure_logger
from utils.pca import GroupwisePCA
from utils.persistent_stats import PERS_STATS_NAMES, STATISTIC_NAMES

sklearn.set_config(enable_metadata_routing=True)

__all__ = [
	"get_patient_ids",
	"read_patient_data",
	"run_classification_all_patients",
	"run_classification_single_patient",
	"save_classification_results",
]

SAMPLE_TYPES_ENUM = pl.Enum(["adenoma", "cancer"])


class FeatureParts:
	"""Lazily parse a feature name into its components."""

	dgm_and_dim: str
	_dgm: DiagramName
	_dim: int
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
		self.statistic = tokens[1]
		self.codomain = tokens[2]

	def _parse_dgm_and_dim(self) -> None:
		if hasattr(self, "_dim") and hasattr(self, "_dgm"):
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
		return self._dim

	@property
	def dgm(self) -> DiagramName:
		self._parse_dgm_and_dim()
		return self._dgm


@dataclass(slots=True)
class ClassificationParams:
	"""Parameters for classification."""

	pipeline: Pipeline = field(kw_only=True)
	"""Base model for classification."""
	splitter: StratifiedKFold = field(kw_only=True)
	"""Splitter for the cross-validation diagnostic."""
	num_permutations: int = field(kw_only=True)
	"""Number of random permutations to use to estimate permutation importance scores."""
	permutations_seed: int = field(kw_only=True)
	"""Seed to generate the random permutations for feature importance testing."""
	score_func: ScoreFunc = field(kw_only=True)
	"""Score function to evaluate a prediction."""


class FittedClassifier(Protocol):
	"""A fitted classifier that provides class probabilities."""

	classes_: ArrayLike
	def predict_proba(self, data: ArrayLike) -> np.ndarray:
		"""Predict class probabilities."""
		...


def get_patient_ids(
	*stat_dump_dirs: PathLike,
) -> NDArray1[np.integer]:
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
		logging.getLogger(__name__).debug(
			"Reading data for patient id %s from %s.",
			patient_id,
			path,
		)
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


def get_permutation_importance_scores(
	data: pl.DataFrame,
	y: NDArray1[np.integer],
	*,
	reference_score: float,
	positive_class: int,
	group_columns: Mapping[str, Sequence[str]],
	fitted_classifier: FittedClassifier,
	permutations_seed: int,
	score_func: ScoreFunc,
	num_permutations: int,
) -> dict[str, NDArray1[np.floating]]:
	"""Evaluate grouped permutation importance using a fitted classifer."""
	# Pre-generate the same permutations to use for all groups
	rng = np.random.default_rng(permutations_seed)
	permutations = [rng.permutation(data.height) for _ in range(num_permutations)]

	# Each group importance is a distribution, not a single number
	groups = list(group_columns.keys())
	group_importance_distributions = {
		group: np.empty(num_permutations, dtype=float) for group in groups
	}
	for group in tqdm(
		groups,
		desc="Group",
		leave=False,
	):
		cur_group_columns = group_columns[group]
		for permutation_index, permutation in tqdm(
			enumerate(permutations),
			total=len(permutations),
			desc="Permutation",
			leave=False,
		):
			data_pca_permuted = data.update(
				data.select(cur_group_columns).gather(permutation),
			)
			permuted_predictions = predict_class_probability(
				fitted_classifier,
				data_pca_permuted,
				positive_class,
			)
			permuted_score = score_func(y, permuted_predictions)
			group_importance_distributions[group][permutation_index] = (
				reference_score - permuted_score
			)
	return group_importance_distributions


def predict_class_probability(
	classifier: FittedClassifier,
	data_test: pl.DataFrame,
	class_label: int,
) -> NDArray1[np.floating]:
	"""Predict the probability of specific class using a fitted classifier."""
	classes_: NDArray1[np.integer] = np.asarray(classifier.classes_)
	positive_class_indices = np.flatnonzero(classes_ == class_label)
	if len(positive_class_indices) != 1:
		errmsg = f"Could not identify the positive class in {classes_.tolist()}."
		raise RuntimeError(errmsg)
	return classifier.predict_proba(data_test)[:, positive_class_indices.item()]


def run_classification_single_patient(  # noqa: C901, PLR0915
	data: pl.DataFrame,
	*,
	classification_params: ClassificationParams,
) -> dict:
	"""Run classification for a single patient.

	One stratified K-fold cross-validation pass records the training and test score
	for each fold, along with a score from the concatenated out-of-fold predictions.

	A final model is then fitted on the complete patient dataset. Feature importance
	for each cell group is the distribution of score decreases obtained by jointly
	permuting that group's feature columns and predicting with the fitted final model.
	These importances measure final-model reliance within the observed dataset; they
	do not establish causal effects.
	"""
	# If data is empty return empty dictionary
	if data.height == 0 or data.width == 0:
		return {}

	# Prediction target
	y: NDArray1[np.integer] = (
		data.get_column("sample_type").to_physical().to_numpy()
	)

	# If the target has only samples of a given type, return an empty dictionary
	classes, class_counts = np.unique(y, return_counts=True)
	if len(classes) == 1:
		return {}
	if len(classes) != len(SAMPLE_TYPES_ENUM.categories):
		errmsg = (
			f"Expected {len(SAMPLE_TYPES_ENUM.categories)} sample types, got {classes.tolist()}."
		)
		raise ValueError(errmsg)
	positive_class = classes.max().item()

	# Parameter validation
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
		errmsg = "Got a table with no feature columns."
		raise ValueError(errmsg)
	avg_perc_features = data.count().to_numpy().mean() / data.height
	data = data.fill_null(strategy="zero")

	# Get a mapping of cell groups to column names
	parsed_column_names = [FeatureParts(colname) for colname in data.columns]
	cell_groups = sorted(
		{parsed_column_name.codomain for parsed_column_name in parsed_column_names},
	)
	cell_group_columns: dict[str, list[str]] = {cell_group: [] for cell_group in cell_groups}
	for column, parsed_column_name in zip(data.columns, parsed_column_names, strict=True):
		cell_group = parsed_column_name.codomain
		cell_group_columns[cell_group].append(column)

	# Run one cross-validation pass as an overfitting diagnostic.
	cv_train_scores = np.empty(n_splits, dtype=float)
	cv_test_scores = np.empty(n_splits, dtype=float)
	cv_baseline_test_scores = np.empty(n_splits, dtype=float)
	oof_predictions = np.empty(len(y), dtype=float)
	oof_baseline_predictions = np.empty(len(y), dtype=float)
	cv_splits = classification_params.splitter.split(data, y)
	for fold_index, (train_idxs, test_idxs) in enumerate(
		tqdm(
			cv_splits,
			desc="Cross-validation fold",
			total=n_splits,
			leave=False,
		),
	):
		data_train = data.gather(train_idxs)
		data_test = data.gather(test_idxs)
		estimator = cast("Pipeline", clone(classification_params.pipeline))
		estimator = cast(
			"FittedClassifier",
			estimator.fit(
				data_train,
				y[train_idxs],
				columns_by_group=cell_group_columns,
			),
		)
		train_predictions = predict_class_probability(estimator, data_train, positive_class)
		test_predictions = predict_class_probability(estimator, data_test, positive_class)
		cv_train_scores[fold_index] = classification_params.score_func(
			y[train_idxs],
			train_predictions,
		)
		cv_test_scores[fold_index] = classification_params.score_func(
			y[test_idxs],
			test_predictions,
		)
		oof_predictions[test_idxs] = test_predictions

		baseline_estimator = cast(
			"FittedClassifier",
			DummyClassifier().fit(data_train, y[train_idxs]),
		)
		baseline_predictions = predict_class_probability(
			baseline_estimator,
			data_test,
			positive_class,
		)
		oof_baseline_predictions[test_idxs] = baseline_predictions
		cv_baseline_test_scores[fold_index] = classification_params.score_func(
			y[test_idxs],
			baseline_predictions,
		)
	cv_oof_score = classification_params.score_func(y, oof_predictions)

	final_model = cast("Pipeline", clone(classification_params.pipeline))

	# Optimization: GroupwisePCA is equivariant to permutations in the input rows.
	# So after permuting we needn't compute PCA again.
	# We will compute PCA on the dataset once, and only run permutations post PCA.
	if not isinstance(final_model, Pipeline):
		errmsg = (
			"The current implementation of grouped permutation "
			"importance requires a fitted pipeline."
		)
		raise TypeError(errmsg)

	final_model.fit(data, y, columns_by_group=cell_group_columns)
	fitted_pca = final_model.named_steps.get("pca")

	if not isinstance(fitted_pca, GroupwisePCA):
		errmsg = "Pipeline step 'pca' must be a fitted GroupwisePCA."
		raise TypeError(errmsg)

	fitted_classifier = final_model.named_steps.get("classification")
	if fitted_classifier is None:
		errmsg = "Pipeline is missing the 'classification' step."
		raise ValueError(errmsg)

	# Compute grouped PCA once for the whole dataset
	data_pca = fitted_pca.transform(data)

	# Predictions against the fitted model to use as a baseline score.
	# The decrease in importance after permuting is evaluated
	# against this score.
	final_model_predictions = predict_class_probability(fitted_classifier, data_pca, positive_class)
	final_model_score = classification_params.score_func(y, final_model_predictions)

	pca_columns_by_group = {
		group: group_model.output_names for group, group_model in fitted_pca.models_.items()
	}
	group_importance_distributions = get_permutation_importance_scores(
		data_pca,
		y,
		reference_score=final_model_score,
		positive_class=positive_class,
		group_columns=pca_columns_by_group,
		fitted_classifier=fitted_classifier,
		permutations_seed=classification_params.permutations_seed,
		score_func=classification_params.score_func,
		num_permutations=classification_params.num_permutations,
	)
	return {
		"cv_train_scores": cv_train_scores,
		"cv_test_scores": cv_test_scores,
		"cv_oof_score": cv_oof_score,
		"cv_baseline_test_scores": cv_baseline_test_scores,
		"final_model_score": final_model_score,
		"feature_importances": group_importance_distributions,
		"num_cross_val_splits": n_splits,
		"num_permutations": classification_params.num_permutations,
		"num_samples": data.shape[0],
		"avg_perc_features": avg_perc_features,
	}


def brier_score(target: ArrayLike, pred: ArrayLike) -> float:
	return float(1.0 - brier_score_loss(target, pred))


def save_classification_results(
	classification_results: Sequence[dict],
	output_dir: PathLike,
) -> None:
	"""Write classification results to parquet files."""
	logging.getLogger(__name__).info("Saving classification results to %s", output_dir)
	output_path = Path(output_dir).resolve()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	result_table: pa.Table = pl.from_dicts(classification_results).to_arrow()
	pq.write_to_dataset(
		result_table,
		root_path=str(output_path),
		partition_cols=["patient_id"],
		compression="zstd",
		compression_level=9,
		existing_data_behavior="delete_matching",
	)


def results_already_exist(patient_id: int, output_dir: PathLike) -> bool:
	output_path = Path(output_dir).resolve()
	if not output_path.is_dir():
		return False
	try:
		return not (
			pl.scan_parquet(
				output_path,
				hive_partitioning=True,
				missing_columns="insert",
			)
			.filter(pl.col("patient_id") == patient_id)
			.limit(1)
			.collect()
			.is_empty()
		)
	except pl.exceptions.PolarsError:
		return False


def run_classification_all_patients(
	*,
	stats_dirs: list[PathLike],
	output_dir: PathLike,
	labels_include: tuple[str, ...],
	labels_exclude: tuple[str, ...],
	keep_epithelium: bool,
	cross_val_splits: int,
	num_permutations: int,
	num_workers: int,
	resume: bool,
) -> None:
	"""Run classification for each patient."""
	logger = logging.getLogger(__name__)
	logger.info("Starting classification.")

	patient_ids = get_patient_ids(*stats_dirs)

	for patient_id in tqdm(patient_ids, desc="Patients"):
		if resume and results_already_exist(patient_id.item(), output_dir):
			logger.debug(
				"Skipping classification for patient_id: %s",
				patient_id,
			)
			continue

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
			n_jobs=max(1, num_workers),
			max_samples=1.0,
			bootstrap=False,
			random_state=0,
		)
		classifier = Pipeline(
			[
				(
					"pca",
					GroupwisePCA().set_fit_request(columns_by_group=True),
				),
				("classification", ensemble_classifier),
			],
		)
		splitter = StratifiedKFold(
			n_splits=cross_val_splits,
			shuffle=True,
			random_state=0,
		)

		classification_params = ClassificationParams(
			pipeline=classifier,
			splitter=splitter,
			num_permutations=num_permutations,
			permutations_seed=0,
			score_func=brier_score,
		)
		result = None
		try:
			result = run_classification_single_patient(
				data,
				classification_params=classification_params,
			)
		except Exception:
			errmsg = f"Encountered error during classification for patient {patient_id.item()}."
			logger.exception(errmsg)

		if isinstance(result, dict) and len(result) > 0:
			result = {"patient_id": patient_id.item()} | result
			save_classification_results([result], output_dir)


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
		"--output-dir": {
			"type": str,
			"default": str(base_path.joinpath("results", "classification_results")),
			"help": "Absolute or relative path to the output directory to save.",
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
			"help": "Number of folds in the stratified cross-validation diagnostic.",
		},
		"--num-permutations": {
			"type": int,
			"default": 30,
			"help": (
				"Number of permutations to use for computing "
				"grouped permutation importance scores. "
				"Each feature group is jointly permuted this many times and evaluated "
				"using a model fitted once on the complete patient dataset."
			),
		},
		"--num-workers": {
			"default": 0,
			"type": int,
			"help": (
				"Number of CPUs to use for parallel processing. Set to 0 to use all available CPUs."
			),
		},
		"--logfile-dir": {
			"default": str(base_path.joinpath("logs")),
			"type": str,
			"help": "Directory where log files will be saved.",
		},
		"--resume": {
			"action": argparse.BooleanOptionalAction,
			"help": (
				"If --resume is set, don't recompute classification results from previous runs."
			),
			"default": True,
		},
		"--verbosity": {
			"default": "INFO",
			"type": str,
			"choices": ("DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"),
			"help": (
				"Set the verbosity level of the logger. "
				"Only messages of this level and higher will be logged."
			),
		},
	}

	for k, v in default_options.items():
		_ = parser.add_argument(k, **v)
	args = parser.parse_args()
	if args.num_workers == 0:
		cpu_count = os.cpu_count()
		args.num_workers = cpu_count - 1 if cpu_count is not None else 1
	logger = _init_logging(args)
	logger.debug("Provided arguments:\n%s", pprint.pformat(vars(args)))
	labels_include = args.labels_include or ()
	labels_exclude = args.labels_exclude or ()
	Path(args.output_dir).resolve().mkdir(parents=True, exist_ok=True)
	classification_results = run_classification_all_patients(
		stats_dirs=args.stats_dirs,
		labels_include=labels_include,
		labels_exclude=labels_exclude,
		keep_epithelium=args.keep_epithelium,
		num_workers=args.num_workers,
		num_permutations=args.num_permutations,
		cross_val_splits=args.cross_val_splits,
		output_dir=args.output_dir,
		resume=args.resume,
	)

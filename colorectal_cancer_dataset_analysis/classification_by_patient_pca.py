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
from typing import TYPE_CHECKING, ClassVar, TypeGuard, cast, get_args

import numpy as np
import polars as pl
import polars.selectors as cs
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn
from chalc.sixpack.types import DiagramName
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm

if TYPE_CHECKING:
	from collections.abc import Callable, Sequence

	from numpy.typing import ArrayLike

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


@dataclass(slots=True)
class ClassificationParams:
	"""Parameters for classification."""

	classifier: BaseEstimator = field(kw_only=True)
	"""Base model for classification."""
	splitter: StratifiedKFold = field(kw_only=True)
	"""Splitter for the cross-validation diagnostic."""
	num_permutations: int = field(kw_only=True)
	"""Number of random permutations to use to estimate permutation importance scores."""
	permutations_seed: int = field(kw_only=True)
	"""Seed to generate the random permutations for feature importance testing."""
	score_func: Callable[[ArrayLike, ArrayLike], float] = field(kw_only=True)
	"""Score function to evaluate a prediction."""


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
	cell_groups = sorted({FeatureParts(colname).codomain for colname in data.columns})
	cell_group_columns = {cell_group: [] for cell_group in cell_groups}
	for column in data.columns:
		cell_group = FeatureParts(column).codomain
		cell_group_columns[cell_group].append(column)

	def predict_positive_probability(
		estimator: BaseEstimator,
		data_test: pl.DataFrame,
	) -> np.ndarray:
		classes_: np.ndarray[tuple[int], np.dtype[np.integer]] = np.asarray(estimator.classes_)
		positive_class_indices = np.flatnonzero(classes_ == classes.max())
		if len(positive_class_indices) != 1:
			errmsg = f"Could not identify the positive class in {classes_.tolist()}."
			raise RuntimeError(errmsg)
		return estimator.predict_proba(data_test)[:, positive_class_indices.item()]

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
		estimator = clone(classification_params.classifier)
		estimator.fit(
			data_train,
			y[train_idxs],
			columns_by_group=cell_group_columns,
		)
		train_predictions = predict_positive_probability(estimator, data_train)
		test_predictions = predict_positive_probability(estimator, data_test)
		cv_train_scores[fold_index] = classification_params.score_func(
			y[train_idxs],
			train_predictions,
		)
		cv_test_scores[fold_index] = classification_params.score_func(
			y[test_idxs],
			test_predictions,
		)
		oof_predictions[test_idxs] = test_predictions

		baseline_estimator = DummyClassifier().fit(data_train, y[train_idxs])
		baseline_predictions = predict_positive_probability(baseline_estimator, data_test)
		oof_baseline_predictions[test_idxs] = baseline_predictions
		cv_baseline_test_scores[fold_index] = classification_params.score_func(
			y[test_idxs],
			baseline_predictions,
		)
	cv_oof_score = classification_params.score_func(y, oof_predictions)

	# Fit one final model to all observations and evaluate grouped permutation importance.
	final_model = clone(classification_params.classifier)
	final_model.fit(data, y, columns_by_group=cell_group_columns)
	final_model_predictions = predict_positive_probability(final_model, data)
	final_model_score = classification_params.score_func(y, final_model_predictions)
	rng = np.random.default_rng(classification_params.permutations_seed)
	permutations = [
		rng.permutation(data.height) for _ in range(classification_params.num_permutations)
	]
	group_importance_distributions = {
		cell_group: np.empty(classification_params.num_permutations, dtype=float)
		for cell_group in cell_groups
	}
	for cell_group in tqdm(
		cell_groups,
		desc="Cell group",
		leave=False,
	):
		group_columns = cell_group_columns[cell_group]
		for permutation_index, permutation in tqdm(
			enumerate(permutations),
			desc="Permutation",
			leave=False,
		):
			data_permuted = data.update(
				data.select(group_columns).gather(permutation),
			)
			permuted_predictions = predict_positive_probability(final_model, data_permuted)
			permuted_score = classification_params.score_func(y, permuted_predictions)
			group_importance_distributions[cell_group][permutation_index] = (
				final_model_score - permuted_score
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
			classifier=classifier,
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

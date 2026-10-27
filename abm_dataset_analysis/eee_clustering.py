#! /usr/bin/env python
"""Cluster ABM spatial signatures and score grouped permutation refits against EEE labels."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, cast

import numpy as np
import polars as pl
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, rand_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from utils.logging import configure_logger
from utils.read_stats import read_feature_data
from utils.stat_schema import DiagramName, FeatureParts

if TYPE_CHECKING:
	from collections.abc import Callable, Mapping, Sequence
	from os import PathLike

	from numpy.typing import ArrayLike

METADATA_COLUMNS = ("ID", "M2_ratio", "c_half", "chi_c^m", "label")
type FeatureGroup = tuple[str, DiagramName, int]
type ClusteringMetric = Literal["ari", "nmi", "binder_loss"]

# Importance of a feature is positive when permutation of columns from that feature
# worsens agreement with ground truth annotations.
SCORE_DIRECTIONS: dict[ClusteringMetric, int] = {"ari": 1, "nmi": 1, "binder_loss": -1}


def clustering_scores(y: ArrayLike, predictions: ArrayLike) -> dict[ClusteringMetric, float]:
	"""Evaluate a partition against EEE labels using ARI, NMI, and Binder loss.

	NMI uses the arithmetic entropy normalizer. Binder loss uses equal split/merge
	weights and is normalized by the number of unordered sample pairs: the fraction
	of pairs whose co-clustering status differs, equivalently 1 minus Rand index.
	It is a loss (zero is best), and is not adjusted for chance.
	"""
	return {
		"ari": adjusted_rand_score(y, predictions),
		"nmi": float(normalized_mutual_info_score(y, predictions, average_method="arithmetic")),
		"binder_loss": float(1.0 - rand_score(y, predictions)),
	}


class FittedClusterer(Protocol):
	"""A fitted baseline estimator that supports scikit-learn cloning and refitting."""

	def get_params(self, *, deep: bool = True) -> dict[str, object]:
		"""Return the parameters needed to construct an unfitted clone."""
		...

	def predict(self, data: np.ndarray, /) -> ArrayLike:
		"""Assign each observation to a fitted cluster."""
		...

	def fit_predict(self, data: np.ndarray, /) -> ArrayLike:
		"""Fit the estimator and return its learned partition."""
		...


def read_all_data(
	stats_dir: PathLike,
	labels_include: tuple[str, ...] = (),
	labels_exclude: tuple[str, ...] = (),
) -> pl.DataFrame:
	"""Use the same filtering and pivot as colorectal classification."""
	return read_feature_data(
		stat_dump_dirs=[stats_dir],
		index_columns=METADATA_COLUMNS,
		labels_include=labels_include,
		labels_exclude=labels_exclude,
	)


def group_feature_columns(columns: Sequence[str]) -> dict[FeatureGroup, list[str]]:
	"""Collect all statistics from the same cell group, diagram, and degree."""
	groups: dict[FeatureGroup, list[str]] = {}
	for column in sorted(columns):
		parts = FeatureParts(column)
		key = (parts.codomain, parts.dgm, parts.dim)
		groups.setdefault(key, []).append(column)
	return groups


def get_permutation_importance_scores(
	data: pl.DataFrame,
	y: ArrayLike,
	*,
	fitted_clusterer: FittedClusterer,
	group_columns: Mapping[FeatureGroup, Sequence[str]],
	num_permutations: int,
	permutations_seed: int,
	disable_progressbar: bool = False,
) -> dict[FeatureGroup, dict[ClusteringMetric, np.ndarray]]:
	"""Return grouped importance after refitting clustering on each permutation.

	ARI/NMI importance is baseline minus permuted score. Binder importance is
	permuted minus baseline loss, so positive values always indicate deterioration.

	A single row permutation is applied to all columns in a group, including
	missing values. The same permutations are reused across groups. Each permuted
	dataset is fitted with a fresh clone of the complete clustering pipeline,
	including imputation and scaling, preserving its parameters and random seed.
	The fitted baseline estimator is unchanged. EEE annotations are used only to
	score the resulting partitions. Negative importance values are kept.
	"""
	if num_permutations < 1:
		msg = "num_permutations must be at least 1."
		raise ValueError(msg)
	if data.height < 2 or len(np.asarray(y)) != data.height:  # noqa: PLR2004
		msg = "Expected matching data and EEE annotations for at least two samples."
		raise ValueError(msg)
	if not group_columns:
		msg = "At least one feature group is required."
		raise ValueError(msg)
	flat_columns = [column for columns in group_columns.values() for column in columns]
	if len(flat_columns) != len(set(flat_columns)) or set(flat_columns) != set(data.columns):
		msg = "Feature groups must partition the feature columns exactly once."
		raise ValueError(msg)
	if any(not columns for columns in group_columns.values()):
		msg = "Feature groups must not be empty."
		raise ValueError(msg)
	# clone supports get_params-based estimators, but its stubs require BaseEstimator.
	clone_clusterer = cast("Callable[[FittedClusterer], FittedClusterer]", clone)
	reference_scores = clustering_scores(y, fitted_clusterer.predict(data.to_numpy()))
	rng = np.random.default_rng(permutations_seed)
	permutations = [rng.permutation(data.height) for _ in range(num_permutations)]
	importances = {}
	for group, columns in tqdm(
		group_columns.items(),
		desc="Feature group",
		disable=disable_progressbar,
	):
		differences = {
			metric: np.empty(num_permutations, dtype=float) for metric in SCORE_DIRECTIONS
		}
		for index, permutation in enumerate(permutations):
			# with_columns replaces nulls too; update() would retain the original
			# values at rows whose permuted replacement is null.
			permuted = data.with_columns(data.select(columns).gather(permutation).get_columns())
			permuted_clusterer = clone_clusterer(fitted_clusterer)
			predictions = permuted_clusterer.fit_predict(permuted.to_numpy())
			permuted_scores = clustering_scores(y, predictions)
			for metric, direction in SCORE_DIRECTIONS.items():
				differences[metric][index] = direction * (
					reference_scores[metric] - permuted_scores[metric]
				)
		importances[group] = differences
	return importances


def summarize_parameter_groups(data: pl.DataFrame, predictions: ArrayLike) -> list[dict]:
	"""Report the dominant predicted cluster and purity for each ABM parameter pair."""
	predicted = data.with_columns(pl.Series("predicted_class", np.asarray(predictions)))
	results = []
	for group in predicted.partition_by(["c_half", "chi_c^m"], maintain_order=True):
		labels = group["label"].unique()
		if len(labels) != 1:
			msg = "EEE annotations must agree within each ABM parameter pair."
			raise ValueError(msg)
		clusters, counts = np.unique(group["predicted_class"].to_numpy(), return_counts=True)
		mode = int(np.argmax(counts))
		results.append(
			{
				"c_half": group["c_half"][0],
				"chi_c^m": group["chi_c^m"][0],
				"label": labels[0],
				"predicted_class": int(clusters[mode]),
				"purity": float(counts[mode] / group.height),
			},
		)
	return results


def run_clustering(
	*,
	stats_dir: PathLike,
	labels_include: tuple[str, ...] = (),
	labels_exclude: tuple[str, ...] = (),
	k: int = 3,
	num_permutations: int = 100,
	permutations_seed: int = 0,
	clustering_seed: int = 0,
	standardize: bool = True,
	disable_progressbar: bool = False,
) -> dict:
	"""Fit KMeans and refit each grouped permutation before scoring against EEE labels.

	EEE annotations enter scoring only. Importances compare the original partition
	with partitions learned after shuffling each feature group, using the same
	clustering settings and seed. They describe the observed dataset, not held-out
	performance or causal effects. Each sample is an individual simulation.
	"""
	if num_permutations < 1:
		msg = "num_permutations must be at least 1."
		raise ValueError(msg)
	data = read_all_data(stats_dir, labels_include, labels_exclude)
	if data.height < 2:  # noqa: PLR2004
		msg = "At least two simulation samples are required."
		raise ValueError(msg)
	if not 2 <= k <= data.height:  # noqa: PLR2004
		msg = "k must be between 2 and the number of simulation samples."
		raise ValueError(msg)
	if data["label"].null_count() or data["label"].n_unique() < 2:  # noqa: PLR2004
		msg = "Partition evaluation requires nonmissing EEE annotations with at least two regimes."
		raise ValueError(msg)
	if data.select("c_half", "chi_c^m").null_count().to_numpy().any():
		msg = "ABM parameter values must not be missing."
		raise ValueError(msg)
	if (
		data.group_by("c_half", "chi_c^m")
		.agg(pl.col("label").n_unique())
		.filter(pl.col("label") != 1)
		.height
	):
		msg = "EEE annotations must agree within each ABM parameter pair."
		raise ValueError(msg)
	features = data.drop(METADATA_COLUMNS).rechunk()
	if features.width == 0:
		msg = "No feature columns remain after filtering."
		raise ValueError(msg)
	if features.fill_null(0.0).unique().height < k:
		# Edge case - all samples have the same feature vector.
		# This should never happen.
		msg = "Every sample has the same feature vector - something gone wrong."
		raise ValueError(msg)
	if np.isinf(features.to_numpy().astype(float)).any():
		msg = "Feature values must be finite or missing."
		raise ValueError(msg)

	group_columns = group_feature_columns(features.columns)
	y = data["label"].to_numpy()
	pipeline = Pipeline(
		[
			(
				"imputation",
				SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True),
			),
			("standardization", StandardScaler() if standardize else "passthrough"),
			(
				"clustering",
				KMeans(
					n_clusters=k,
					n_init=20,
					random_state=clustering_seed,
					max_iter=100000,
					algorithm="elkan",
				),
			),
		],
	)
	logging.getLogger(__name__).info(
		"Fitting KMeans to %d samples and %d features.",
		features.height,
		features.width,
	)
	# EEE annotations and simulation/parameter metadata never enter model fitting.
	pipeline.fit(features.to_numpy())
	predictions = np.asarray(pipeline.predict(features.to_numpy()))
	final_scores = clustering_scores(y, predictions)
	distributions = get_permutation_importance_scores(
		features,
		y,
		fitted_clusterer=pipeline,
		group_columns=group_columns,
		num_permutations=num_permutations,
		permutations_seed=permutations_seed,
		disable_progressbar=disable_progressbar,
	)
	importance_records = [
		{
			"cell_group": cell_group.split("/"),
			"diagram": diagram,
			"dimension": dimension,
			"columns": group_columns[(cell_group, diagram, dimension)],
			"importances": {metric: values.tolist() for metric, values in metrics.items()},
			"mean": {metric: float(values.mean()) for metric, values in metrics.items()},
			"std": {metric: float(values.std()) for metric, values in metrics.items()},
		}
		for (cell_group, diagram, dimension), metrics in distributions.items()
	]
	return {
		"labels": predictions.tolist(),
		"sample_ids": data["ID"].to_list(),
		"results": summarize_parameter_groups(data, predictions),
		"final_model_scores": final_scores,
		"feature_importances_for_ground_truth": importance_records,
		"permutation_importance_method": "refit_kmeans",
		"num_permutations": num_permutations,
		"permutations_seed": permutations_seed,
		"clustering_seed": clustering_seed,
		"standardize": standardize,
		"k": k,
		"num_samples": features.height,
		"num_features": features.width,
		"avg_perc_features": float(features.count().to_numpy().mean() / features.height),
		"kmeans_params": pipeline.named_steps["clustering"].get_params(),
	}


def make_parser() -> argparse.ArgumentParser:
	"""Build the ABM clustering CLI, without PCA options."""
	base_path = Path(__file__).parent
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--stats-dir", type=Path, default=base_path / "stats")
	parser.add_argument(
		"--output-file",
		type=Path,
		default=base_path / "results/eee_prediction_results.json",
	)
	cell_types = (
		"M1_Macrophage",
		"M2_Macrophage",
		"Macrophage",
		"Tumour",
		"Vessel",
		"Necrotic",
		"Stroma",
	)
	parser.add_argument("--labels-include", nargs="*", choices=cell_types, default=[])
	parser.add_argument("--labels-exclude", nargs="*", choices=cell_types, default=[])
	parser.add_argument("--k", type=int, default=3)
	parser.add_argument("--num-permutations", type=int, default=100)
	parser.add_argument("--permutations-seed", type=int, default=0)
	parser.add_argument("--clustering-seed", type=int, default=0)
	parser.add_argument(
		"--standardize",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="Standardize each statistic before KMeans; no dimensionality reduction is applied.",
	)
	parser.add_argument("--disable-progressbar", action="store_true")
	parser.add_argument("--logfile-dir", type=Path, default=base_path / "logs")
	parser.add_argument(
		"--verbosity",
		choices=("DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"),
		default="INFO",
	)
	return parser


def main() -> None:
	"""Run clustering and save a new JSON result without overwriting an existing file."""
	args = make_parser().parse_args()
	if args.output_file.exists():
		msg = f"Output already exists; choose a new path: {args.output_file}"
		raise FileExistsError(msg)
	args.logfile_dir.mkdir(parents=True, exist_ok=True)
	configure_logger(
		logging.getLogger(),
		filename=args.logfile_dir / "eee_clustering.log",
		file_level=logging.DEBUG,
		console_level=args.verbosity,
	)
	results = run_clustering(
		stats_dir=args.stats_dir,
		labels_include=tuple(args.labels_include),
		labels_exclude=tuple(args.labels_exclude),
		k=args.k,
		num_permutations=args.num_permutations,
		permutations_seed=args.permutations_seed,
		clustering_seed=args.clustering_seed,
		standardize=args.standardize,
		disable_progressbar=args.disable_progressbar,
	)
	args.output_file.parent.mkdir(parents=True, exist_ok=True)
	with args.output_file.open("x") as file:
		json.dump(results, file, indent=4, allow_nan=False)


if __name__ == "__main__":
	main()

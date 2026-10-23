"""Read persisted statistics and prepare the shared analysis feature table."""

from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import Literal, overload

import polars as pl
import polars.selectors as cs
import pyarrow as pa
import pyarrow.parquet as pq

from .persistent_stats import PERS_STATS_NAMES, STATISTIC_NAMES
from .stat_schema import FeatureParts

__all__ = ["read_feature_data", "read_stats"]


@overload
def read_stats(
	stat_dir: PathLike,
	table_format: Literal["parquet"] = "parquet",
	**kwargs,
) -> pq.ParquetDataset: ...


@overload
def read_stats(
	stat_dir: PathLike,
	table_format: Literal["pyarrow"],
	**kwargs,
) -> pa.Table: ...


@overload
def read_stats(
	stat_dir: PathLike,
	table_format: Literal["polars"],
	**kwargs,
) -> pl.DataFrame: ...


def read_stats(
	stat_dir: PathLike,
	table_format: Literal["parquet", "pyarrow", "polars"] = "parquet",
	**kwargs,
) -> pq.ParquetDataset | pa.Table | pl.DataFrame:
	"""Read a statistics dataset as a Parquet dataset, Arrow table, or Polars frame.

	Additional keyword arguments are passed to ``pyarrow.parquet.ParquetDataset``.
	"""
	if table_format not in ("parquet", "pyarrow", "polars"):
		msg = "Invalid format specified. Use 'parquet', 'pyarrow', or 'polars'."
		raise ValueError(msg)
	dataset = pq.ParquetDataset(Path(stat_dir).resolve(), **kwargs)
	if table_format == "parquet":
		return dataset
	table = dataset.read()
	if table_format == "pyarrow":
		return table
	return pl.DataFrame(table)


def _feature_selectors() -> tuple[cs.Selector, cs.Selector, cs.Selector]:
	"""Select statistics and exclusions before and after the codomain pivot."""
	diagram = r"(ker|cok|dom|cod|im|rel)"
	persistent_statistic = rf"({'|'.join(PERS_STATS_NAMES)})"
	summary_func = rf"({'|'.join(STATISTIC_NAMES)})"
	dgm_statistics = cs.matches(rf"^{diagram}\d+-{persistent_statistic}$")
	pre_pivot_drop = (
		cs.matches(rf"^(cod|rel)\d+-{persistent_statistic}$")
		| cs.matches(rf"^cok0-{persistent_statistic}$")
		| cs.matches(rf"^(dom|im)0-{summary_func}_(birth|midpt|length)$")
		| cs.matches(rf"^{diagram}\d+-iqr")
	)
	singles = cs.matches(rf"^{diagram}\d+-{persistent_statistic}-[^/]+$")
	single_domain = cs.matches(rf"^dom\d+-{persistent_statistic}-[^/]+$")
	multiple_domain = cs.matches(rf"^dom\d+-{persistent_statistic}-[^/]+/.+$")
	return dgm_statistics, pre_pivot_drop, (singles - single_domain) | multiple_domain


def read_feature_data(
	*,
	stat_dump_dirs: Sequence[PathLike],
	index_columns: Sequence[str],
	labels_include: tuple[str, ...] = (),
	labels_exclude: tuple[str, ...] = (),
	row_filter: pl.Expr | None = None,
	column_dtypes: pl.Schema | None = None,
) -> pl.DataFrame:
	"""Prepare features identically for colorectal classification and ABM clustering.

	Select and filter diagrams before pivoting, filter cell groups, sort their names,
	and pivot to one row per sample. Metadata remain explicit index columns. Duplicate
	sample/cell-group rows raise a Polars error rather than silently averaging runs.
	Missing feature values remain null for the caller's imputation step.
	"""
	if not stat_dump_dirs:
		msg = "At least one statistics directory is required."
		raise ValueError(msg)
	if not index_columns:
		msg = "At least one sample index column is required."
		raise ValueError(msg)
	statistics, pre_drop, post_drop = _feature_selectors()
	frames = []
	for path in stat_dump_dirs:
		frame = pl.scan_parquet(
			Path(path).resolve(),
			hive_partitioning=True,
			missing_columns="insert",
		)
		if row_filter is not None:
			frame = frame.filter(row_filter)
		frame = frame.select(*index_columns, "codomain", statistics - pre_drop).drop_nulls(
			"codomain",
		)  # will also drop "filtration_algorithm" and "patient_id"
		if labels_include:
			frame = frame.filter(
				pl.col("codomain").list.set_difference(labels_include).list.len() == 0,
			)
		if labels_exclude:
			frame = frame.filter(
				pl.col("codomain").list.set_intersection(labels_exclude).list.len() == 0,
			)
		if column_dtypes:
			frame = frame.cast(column_dtypes)
		frames.append(frame)
	data = (
		pl.concat(frames, how="diagonal_relaxed", rechunk=True)
		.with_columns(pl.col("codomain").list.sort().list.join("/"))
		.collect()
		.with_columns(cs.float().fill_nan(None))
	)
	feature_columns = [c for c in data.columns if c not in (*index_columns, "codomain")]
	if not feature_columns:
		# If there aren't any features then something has gone wrong
		msg = (
			"Statistics table has no features. "
			"This could be either due to `labels_include - labels_exclude` being empty, "
			"or due to an invalid dataset."
		)
		raise RuntimeError(msg)
	result = data.pivot(
		on="codomain",
		index=list(index_columns),
		values=feature_columns,
		separator="-",
	)
	# Polars omits the value-column prefix for a pivot with just one value column.
	if len(feature_columns) == 1:
		result = result.rename(
			{c: f"{feature_columns[0]}-{c}" for c in result.columns if c not in index_columns},
		)
	result = result.drop(post_drop)
	return result.select(
		*index_columns,
		*sorted(c for c in result.columns if c not in index_columns),
	).sort(index_columns)


def stat_name_split(name: str) -> tuple[str, int, str]:
	"""Split '<diagram><dimension>-<statistic>'."""
	parts = FeatureParts(
		f"{name}-_",  # FeatureParts expects the cell group, so we pass '_' as a dummy
	)
	return parts.dgm, parts.dim, parts.statistic


# Legacy - only used for tests and to document the code.
def discard_feature(feature_name: str) -> bool:
	"""Apply the shared feature rules to a fully qualified feature name."""
	parts = FeatureParts(feature_name)
	return (
		parts.dgm in ("cod", "rel")
		or (parts.dgm == "cok" and parts.dim == 0)
		or (
			parts.dgm in ("dom", "im")
			and parts.dim == 0
			and parts.statistic.rsplit("_", maxsplit=1)[-1] in ("birth", "midpt", "length")
		)
		or parts.statistic.startswith("iqr")
		or ("/" not in parts.codomain and parts.dgm != "dom")
		or ("/" in parts.codomain and parts.dgm == "dom")
	)

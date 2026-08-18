"""Utilities for groupwise PCA."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self, cast, overload

import numpy as np
import polars as pl
from scipy.linalg import svd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted, validate_data

if TYPE_CHECKING:
	from collections.abc import Sequence
	from typing import Literal

	from numpy.typing import ArrayLike

__all__ = ["CustomPCA", "GroupwisePCA"]


class CustomPCA(BaseEstimator):
	"""PCA with custom lapack driver."""

	def __init__(
		self,
		*,
		max_components: int = 1,
		centre: bool = True,
		lapack_driver: Literal["gesdd", "gesvd"] = "gesdd",
	) -> None:
		"""Initialize the estimator.

		If `centre=False`, then the estimator expects the input data
		for `fit` and `transform` to be pre-centred.
		"""
		self.max_components = max_components
		self.centre = centre
		self.lapack_driver = lapack_driver

	def _validate_params(self) -> None:
		if (
			isinstance(self.max_components, (bool, np.bool_))
			or not isinstance(self.max_components, (int, np.integer))
			or self.max_components < 1
		):
			errmsg = f"max_components should be a positive integer, got {self.max_components}."
			raise ValueError(errmsg)

		if self.lapack_driver not in ("gesdd", "gesvd"):
			errmsg = f"Invalid lapack_driver: {self.lapack_driver} not in ('gesdd', 'gesvd')."
			raise ValueError(errmsg)

	def _fit_svd(
		self,
		data: ArrayLike,
	) -> tuple[np.ndarray, np.ndarray]:
		self._validate_params()
		data_array = cast(
			"np.ndarray",
			validate_data(
				self,
				X=data,  # pyright: ignore[reportArgumentType]
				reset=True,
				dtype="numeric",
			),
		)

		if self.centre:
			self.data_mean_ = np.mean(data_array, axis=0)
		else:
			self.data_mean_ = np.zeros(data_array.shape[1], dtype=data_array.dtype)
		data_centred = data_array - self.data_mean_
		# Suppose X is a matrix of zero-mean data,
		# where each row is a sample: X.shape = (n_samples, n_features).
		# The covariance matrix is given by Cov(X) = X^t X.
		# The eigenvectors of Cov(X) are the principal components.
		# Suppose Cov(X) = W D W^t and X = U S V^t by SVD.
		# Then W D W^t = V S U^t U S V^t = V S^2 V^t, so in fact W = V and D = S^2.
		# To project arbitrary data Z onto these principal components,
		# we compute (V^t Z^t)^t = Z V. So we need to store V.
		# Return u and s so that that
		# fit_transform()
		# can directly use them.
		# If we want the projection of the
		# training data onto its principal components, then we can
		# avoid an extra matrix multiplication associated with
		# fit().transform(). We simply want X V = U S V^t V = U S.
		(u, s, vh) = svd(
			data_centred,
			lapack_driver=self.lapack_driver,
			full_matrices=False,
			overwrite_a=False,
			compute_uv=True,
		)
		self.components_ = vh[: self.max_components, :]
		self.n_components_ = self.components_.shape[0]
		return u, s

	def fit(self, data: ArrayLike, _y: ArrayLike | None = None) -> Self:
		"""Compute the principal components from a data matrix.

		It is assumed that columns of the input are features and rows are samples.
		"""
		self._fit_svd(data)
		return self

	def transform(self, data: ArrayLike, _y: ArrayLike | None = None) -> ArrayLike:
		"""Project data points onto the fitted principal components."""
		check_is_fitted(self, ("components_", "n_components_", "data_mean_"))
		data_array = cast(
			"np.ndarray",
			validate_data(
				self,
				data,  # pyright:ignore[reportArgumentType]
				reset=False,
				dtype="numeric",
			),
		)
		return (data_array - self.data_mean_) @ self.components_.T

	def fit_transform(self, data: ArrayLike, _y: ArrayLike | None = None) -> ArrayLike:
		"""Find the principal components and project data onto them."""
		u, s = self._fit_svd(data)
		return u[:, : self.n_components_] * s[: self.n_components_]


class GroupwisePCA(BaseEstimator):
	"""Run PCA on the features from each group separately."""

	n_features_in_: int
	"""Number of features passed in during fitting."""
	feature_names_in_: Sequence[str]
	"""Names of input features seen during fitting."""
	feature_names_out_: Sequence[str]
	"""Names of output features after PCA."""
	models_: dict[str, GroupModel]
	"""Mapping of group names to pipeline and metadata."""

	@dataclass(slots=True)
	class GroupModel:
		"""Pipeline and metadata for a single group of features."""

		columns: Sequence[str]
		"""Columns of the input associated to this feature."""
		pipeline: Pipeline
		"""Pipeline to process this group with."""
		n_components: int
		"""Number of fitted PCA components for this group."""
		output_names: Sequence[str]
		"""Output features names for this group."""

	def __init__(self) -> None:
		"""Initialize the estimator."""

	@overload
	def _fit_groups(
		self,
		data: pl.DataFrame,
		*,
		columns_by_group: dict[str, Sequence[str]] | None,
		return_transformed: Literal[True],
	) -> list[pl.DataFrame]: ...
	@overload
	def _fit_groups(
		self,
		data: pl.DataFrame,
		*,
		columns_by_group: dict[str, Sequence[str]] | None,
		return_transformed: Literal[False],
	) -> None: ...
	@overload
	def _fit_groups(
		self,
		data: pl.DataFrame,
		*,
		columns_by_group: dict[str, Sequence[str]] | None,
		return_transformed: bool,
	) -> list[pl.DataFrame] | None: ...

	def _fit_groups(
		self,
		data: pl.DataFrame,
		*,
		columns_by_group: dict[str, Sequence[str]] | None,
		return_transformed: bool,
	) -> list[pl.DataFrame] | None:
		"""Inner helper for fit and transform methods."""
		if not columns_by_group:
			errmsg = "GroupwisePCA.fit requires non-empty columns_by_group metadata."
			raise ValueError(errmsg)

		_ = validate_data(
			self,
			data,  # pyright:ignore[reportArgumentType]
			reset=True,
			skip_check_array=True,
		)
		self.models_ = {}
		self.feature_names_out_ = []

		results = []

		for group, group_columns in columns_by_group.items():
			columns = tuple(group_columns)
			n_features = len(columns)
			if n_features == 0:
				errmsg = f"Group {group!r} has no feature columns."
				raise ValueError(errmsg)

			# Keep at least five components, or 5% for larger feature groups.
			max_components = max(
				math.ceil(0.05 * n_features),
				min(n_features, 5),
			)

			pipeline = Pipeline(
				[
					(
						"standardization",
						StandardScaler(),
					),
					(
						"pca",
						CustomPCA(
							max_components=max_components,
							centre=False,
							lapack_driver="gesvd",
						),
					),
				],
			)

			group_data = data.select(columns).rechunk()
			group_result_array = None

			if return_transformed:
				group_result_array = pipeline.fit_transform(group_data)
			else:
				pipeline.fit(group_data)

			n_components: int = pipeline.named_steps["pca"].n_components_
			group_output_columns = tuple(
				f"pca_{component}-{group}" for component in range(n_components)
			)

			if return_transformed:
				results.append(pl.DataFrame(group_result_array, schema=group_output_columns))

			self.models_[group] = self.GroupModel(
				columns=columns,
				pipeline=pipeline,
				n_components=n_components,
				output_names=group_output_columns,
			)
			self.feature_names_out_ += group_output_columns

		if return_transformed:
			return results

		return None

	def fit(
		self,
		data: pl.DataFrame,
		_y: ArrayLike | None = None,
		*,
		columns_by_group: dict[str, Sequence[str]] | None = None,
	) -> Self:
		"""Compute the PCA projection operators of each group from input data."""
		self._fit_groups(
			data,
			columns_by_group=columns_by_group,
			return_transformed=False,
		)
		return self

	def transform(self, data: pl.DataFrame) -> pl.DataFrame:
		"""Project each group in the input data onto the computed principal components."""
		check_is_fitted(self, "models_")
		_ = validate_data(
			self,
			data,  # pyright:ignore[reportArgumentType]
			reset=False,
			skip_check_array=True,
		)
		transformed_groups = []

		for model_info in self.models_.values():
			columns = model_info.columns
			pipeline = model_info.pipeline
			output_names = model_info.output_names

			values = pipeline.transform(data.select(columns))

			transformed_groups.append(
				pl.DataFrame(
					values,
					schema=output_names,
				),
			)

		return pl.concat(transformed_groups, how="horizontal").rechunk()

	def fit_transform(
		self,
		data: pl.DataFrame,
		_y: ArrayLike | None = None,
		*,
		columns_by_group: dict[str, Sequence[str]] | None = None,
	) -> pl.DataFrame:
		"""Project each group in the input data onto the corresponding principal components."""
		transformed_groups = self._fit_groups(
			data,
			columns_by_group=columns_by_group,
			return_transformed=True,
		)
		return pl.concat(transformed_groups, how="horizontal").rechunk()

	def get_feature_names_out(
		self,
		input_features: Sequence[str] | None = None,  # ruff: ignore[ARG002]
	) -> np.ndarray:
		"""Return names for the transformed PCA features.

		Argument input_features is ignored.
		"""
		check_is_fitted(self, "feature_names_out_")

		return np.asarray(self.feature_names_out_, dtype=object)

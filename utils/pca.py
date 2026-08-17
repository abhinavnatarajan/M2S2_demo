"""Utilities for groupwise PCA."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Self, cast

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

	def fit(self, data: ArrayLike, _y: ArrayLike | None = None) -> Self:
		"""Compute the principal components from a data matrix.

		It is assumed that columns of the input are features and rows are samples.
		"""
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

		data_array = validate_data(self, X=data, reset=True, dtype="numeric")  # pyright:ignore[reportArgumentType]
		data_array = cast("np.ndarray", data_array)
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
		(_, _, vh) = svd(
			data_centred,
			lapack_driver=self.lapack_driver,
			full_matrices=False,
			overwrite_a=False,
			compute_uv=True,
		)
		self.components_ = vh[: self.max_components, :]
		self.n_components_ = self.components_.shape[0]
		return self

	def transform(self, data: ArrayLike, _y: ArrayLike | None = None) -> ArrayLike:
		"""Project data points onto the fitted principal components."""
		check_is_fitted(self, ("components_", "n_components_", "data_mean_"))
		data_array = validate_data(self, data, reset=False, dtype="numeric")  # pyright:ignore[reportArgumentType]
		data_array = cast("np.ndarray", data_array)
		return (data_array - self.data_mean_) @ self.components_.T


class GroupwisePCA(BaseEstimator):
	"""Run PCA on the features from each group separately."""

	def __init__(self) -> None:
		"""Initialize the estimator."""

	def fit(
		self,
		data: pl.DataFrame,
		_y: ArrayLike | None = None,
		*,
		columns_by_group: dict[str, Sequence[str]] | None = None,
	) -> Self:
		"""Compute the PCA projection operators of each group from input data."""
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
		for group, group_columns in columns_by_group.items():
			columns = list(group_columns)
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
			).fit(data.select(columns))

			self.models_[group] = {
				"columns": columns,
				"pipeline": pipeline,
				"n_components": pipeline.named_steps["pca"].n_components_,
			}

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

		output_names = self.get_feature_names_out()
		name_offset = 0
		for model_info in self.models_.values():
			columns = model_info["columns"]
			pipeline = model_info["pipeline"]
			n_components = model_info["n_components"]

			values = pipeline.transform(data.select(columns))
			new_column_names = output_names[name_offset : name_offset + n_components].tolist()
			name_offset += n_components

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
		y: ArrayLike | None = None,
		*,
		columns_by_group: dict[str, Sequence[str]] | None = None,
	) -> pl.DataFrame:
		"""Project each group in the input data onto the corresponding principal components."""
		return self.fit(data, y, columns_by_group=columns_by_group).transform(data)

	def get_feature_names_out(
		self,
		input_features: Sequence[str] | None = None,
	) -> np.ndarray:
		"""Return names for the transformed PCA features."""
		check_is_fitted(self, ("models_", "n_features_in_"))

		if input_features is not None:
			provided_features = np.asarray(input_features, dtype=object)

			if provided_features.ndim != 1:
				errmsg = "input_features must be a one-dimensional sequence."
				raise ValueError(errmsg)

			if len(provided_features) != self.n_features_in_:
				errmsg = (
					f"input_features has {len(provided_features)} features, "
					f"but GroupwisePCA was fitted with {self.n_features_in_} features."
				)
				raise ValueError(errmsg)

			if hasattr(self, "feature_names_in_") and not np.array_equal(
				provided_features,
				self.feature_names_in_,
			):
				errmsg = "input_features must match feature_names_in_ exactly."
				raise ValueError(errmsg)

		return np.asarray(
			[
				f"pca_{component}-{group}"
				for group, model_info in self.models_.items()
				for component in range(model_info["n_components"])
			],
			dtype=object,
		)

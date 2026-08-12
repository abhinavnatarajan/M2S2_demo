"""Utilities to compute persistent statistics."""

from typing import Literal, Self

import numpy as np
from gudhi.representations.preprocessing import BirthPersistenceTransform, DiagramScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler

type NumpyVector[T: np.generic] = np.ndarray[tuple[int], np.dtype[T]]
type Barcode = np.ndarray[tuple[int, Literal[2]], np.dtype[np.float64]]

__all__ = ["PERS_STATS_NAMES", "get_pers_stats"]


def _bar_cleaner(barcode: Barcode) -> Barcode:
	"""Get rid of the diagonal elements in the barcode."""
	if np.size(barcode) > 0:
		return barcode[barcode[:, 0] != barcode[:, 1]]
	return np.zeros(shape=(0, 2), dtype=np.float64)


# ruff: ignore[N803]
def _automatic_sample_range(
	sample_range: tuple[float, float],
	X: list[Barcode],
	y: None = None,
) -> tuple[float, float]:
	"""Compute sample range from persistence diagrams if one of the sample_range values is nan.

	Parameters
	----------
		sample_range : tuple[float, float]
			Minimum and maximum of all piecewise-linear function domains,
			of the form (x_min, x_max).
		barcodes : list[Barcode]
			Input persistence diagrams.
		barcode_labels : list[str]
			Persistence diagram labels (unused).

	"""
	nan_in_range = np.isnan(sample_range)
	if nan_in_range.any():
		try:
			pre = DiagramScaler(
				use=True,
				scalers=[([0], MinMaxScaler()), ([1], MinMaxScaler())],
			).fit(X, y)
			[mx, _] = [pre.scalers[0][1].data_min_[0], pre.scalers[1][1].data_min_[0]]
			[_, my] = [pre.scalers[0][1].data_max_[0], pre.scalers[1][1].data_max_[0]]
			return tuple(np.where(nan_in_range, np.array([mx, my]), sample_range))
		except ValueError:
			# Empty persistence diagram case - https://github.com/GUDHI/gudhi-devel/issues/507
			pass
	return sample_range


class Entropy(BaseEstimator, TransformerMixin):
	"""This is a class for computing persistence entropy.

	Persistence entropy is a statistic for persistence diagrams
	inspired from Shannon entropy.
	This statistic can also be used to compute a feature vector,
	called the entropy summary function.
	See https://arxiv.org/pdf/1803.08304.pdf for more details.
	Note that a previous implementation was contributed by Manuel Soriano-Trigueros.
	"""

	def __init__(
		self,
		*,
		mode: str = "scalar",
		normalized: bool = True,
		resolution: int = 100,
		sample_range: tuple[float, float] | None = None,
	) -> None:
		"""Construct the Entropy class.

		Parameters
		----------
			mode: str
				What entropy to compute.
				Either "scalar" for computing the entropy statistics,
				or "vector" for computing the entropy summary functions (default "scalar").
			normalized: bool
				Whether to normalize the entropy summary function (default True).
				Used only if **mode** = "vector".
			resolution: int
				Number of samples for the entropy summary function (default 100).
				Used only if **mode** = "vector".
			sample_range: tuple[float, float]
				Minimum and maximum of the
				entropy summary function domain, of the form [x_min, x_max]
				Defaults to [numpy.nan, numpy.nan].
				It is the interval on which samples will be drawn evenly.
				If one of the values is numpy.nan, it can be computed from
				the persistence diagrams with the fit() method.
				Used only if **mode** = "vector".
		"""
		if sample_range is None:
			sample_range = (np.nan, np.nan)
		self.mode, self.normalized, self.resolution, self.sample_range = (
			mode,
			normalized,
			resolution,
			sample_range,
		)

	# ruff: ignore[N803]
	def fit(self, X: list[Barcode], y: None = None) -> Self:
		"""
		Fit the Entropy class on a list of persistence diagrams.

		Parameters
		----------
			X : list of n x 2 numpy arrays
				Input persistence diagrams.
			y : None
				Unused
		"""
		self.sample_range = _automatic_sample_range(self.sample_range, X, y)
		return self

	# ruff: ignore[N803]
	def transform(self, X: list[Barcode]) -> NumpyVector[np.floating]:
		"""Compute the entropy for each persistence diagram and concatenate the results.

		Parameters
		----------
			X : list of n x 2 numpy arrays
				Input persistence diagrams.

		Returns
		-------
			numpy array with shape (number of diagrams) x (1 if **mode** = "scalar" else **resolution**): output entropy.
		"""
		num_diag, x_fit = len(X), []
		x_values = np.linspace(self.sample_range[0], self.sample_range[1], self.resolution)
		step_x = x_values[1] - x_values[0]
		new_x = BirthPersistenceTransform().fit_transform(X)

		for i in range(num_diag):
			orig_diagram, diagram, num_pts_in_diag = X[i], new_x[i], X[i].shape[0]
			try:
				new_diagram = DiagramScaler().fit_transform([diagram])[0]
			except ValueError:
				# Empty persistence diagram case - https://github.com/GUDHI/gudhi-devel/issues/507
				assert len(diagram) == 0
				new_diagram = np.empty(shape=[0, 2])
			p = new_diagram[:, 1]
			# we need this condition to avoid dividing by zero
			if (p != 0).any():
				p = p / np.sum(p)

			# This function is necessary to guarantee that 0*log(0)=0 later
			def log0(x: float) -> float:
				return 0 if x == 0 else np.log(x)

			log0_ufunc = np.frompyfunc(log0, 1, 1)
			if self.mode == "scalar":
				ent = -np.dot(p, log0_ufunc(p))
				x_fit.append(np.array([[ent]]))
			else:
				ent = np.zeros(self.resolution)
				for j in range(num_pts_in_diag):
					[px, py] = orig_diagram[j, :2]
					if px != py:
						min_idx = np.clip(
							np.ceil((px - self.sample_range[0]) / step_x).astype(int),
							0,
							self.resolution,
						)
						max_idx = np.clip(
							np.ceil((py - self.sample_range[0]) / step_x).astype(int),
							0,
							self.resolution,
						)
						ent[min_idx:max_idx] -= p[j] * log0_ufunc(p[j])
				if self.normalized:
					ent = ent / np.linalg.norm(ent, ord=1)
				x_fit.append(np.reshape(ent, [1, -1]))

		return np.concatenate(x_fit, axis=0)

	def __call__(self, diag: Barcode) -> NumpyVector[np.floating]:
		"""Apply Entropy on a single persistence diagram and outputs the result.

		Parameters
		----------
			diag : n x 2 numpy array
				Input persistence diagram.

		Returns
		-------
			numpy array with shape (1 if **mode** = "scalar" else **resolution**): output entropy.
		"""
		return self.fit_transform([diag])[0, :]


# Average of Birth and Death of the barcode
def _births[M: int](bar: Barcode) -> NumpyVector[np.float64]:
	return bar[:, 0]


def _deaths[M: int](bar: Barcode) -> NumpyVector[np.float64]:
	return bar[:, 1]


def _midpts[M: int](bar: Barcode) -> NumpyVector[np.float64]:
	return (bar[:, 0] + bar[:, 1]) / 2


def _lengths[M: int](bar: Barcode) -> NumpyVector[np.float64]:
	return np.abs(bar[:, 1] - bar[:, 0])


def _iqr(x: NumpyVector[np.floating]) -> np.floating:
	return np.subtract(*np.percentile(x, [75, 25]))


def _ran(x: NumpyVector[np.floating]) -> np.floating:
	return np.ptp(x)


def _p10(x: NumpyVector[np.floating]) -> np.floating:
	return np.percentile(x, 10, axis=None)


def _p25(x: NumpyVector[np.floating]) -> np.floating:
	return np.percentile(x, 25, axis=None)


def _p75(x: NumpyVector[np.floating]) -> np.floating:
	return np.percentile(x, 75, axis=None)


def _p90(x: NumpyVector[np.floating]) -> np.floating:
	return np.percentile(x, 90, axis=None)


var_funcs = {
	"birth": _births,
	"death": _deaths,
	"midpt": _midpts,
	"length": _lengths,
}
statistic_funcs = {
	"avg": np.mean,
	"sd": np.std,
	"med": np.median,
	"iqr": _iqr,
	"range": _ran,
	"p25": _p25,
	"p75": _p75,
	"p10": _p10,
	"p90": _p90,
}

PERS_STATS_NAMES = [
	statistic_name + "_" + var_name for var_name in var_funcs for statistic_name in statistic_funcs
] + ["num_bars", "entropy"]
VAR_NAMES = list(var_funcs.keys())
STATISTIC_NAMES = list(statistic_funcs.keys())


def get_pers_stats(
	barcode: Barcode,
) -> np.ndarray[tuple[Literal[38]], np.dtype[np.float64]]:
	"""Compute persistent statistics from a barcode.

	Arguments:
		barcode (Mx2 numpy matrix of floats): Matrix of birth-death pairs.

	Returns:
		Vector of persistent statistics.
	"""
	barcode = _bar_cleaner(barcode)
	finite_bars = barcode[~np.any(~np.isfinite(barcode), axis=1)]
	if len(finite_bars.shape) != 2 or finite_bars.shape[1] != 2:  # ruff: ignore[PLR2004]
		errmsg = f"Got shape: {finite_bars.shape}"
		raise RuntimeError(errmsg)

	if finite_bars.size > 0:
		entropy = Entropy(mode="scalar")(finite_bars).item()
		stats = np.array(
			[
				statistic_func(var_func(finite_bars))
				for var_func in var_funcs.values()
				for statistic_func in statistic_funcs.values()
			]
			+ [len(barcode), entropy],
			dtype=np.float64,
		)
		stats[~np.isfinite(stats)] = 0
	else:
		stats = np.array([0.0] * (len(var_funcs) * len(statistic_funcs) + 2), dtype=np.float64)

	return stats

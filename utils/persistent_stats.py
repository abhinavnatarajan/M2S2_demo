from typing import Literal, TypeVar

import numpy as np
from gudhi.representations.preprocessing import BirthPersistenceTransform, DiagramScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler

M = TypeVar("M", bound=int)
type NumpyVector[M: int, T: np.generic] = np.ndarray[tuple[M], np.dtype[T]]
type Barcode[M: int] = np.ndarray[tuple[M, Literal[2]], np.dtype[np.float64]]

__all__ = ["get_pers_stats", "pers_stats_names"]


def _bar_cleaner(barcode: Barcode[M]) -> Barcode[int]:
    """Get rid of the diagonal elements in the barcode."""
    if np.size(barcode) > 0:
        return barcode[barcode[:, 0] != barcode[:, 1]]
    return np.zeros(shape=(0, 2), dtype=np.float64)


def _automatic_sample_range(sample_range, X, y):
    """Compute sample range from persistence diagrams if one of the sample_range values is nan.

    Parameters
    ----------
            sample_range :
                    minimum and maximum of all piecewise-linear function domains,
                    of the form [x_min, x_max].
            X : list[np.ndarray[tuple[int, Literal[2]], np.dtype[np.float64]]]
                    input persistence diagrams.
            y : np.ndarray[tuple[int], np.dtype[np.str_]]
                    persistence diagram labels (unused).

    """
    nan_in_range = np.isnan(sample_range)
    if nan_in_range.any():
        try:
            pre = DiagramScaler(
                use=True, scalers=[([0], MinMaxScaler()), ([1], MinMaxScaler())]
            ).fit(X, y)
            [mx, _] = [pre.scalers[0][1].data_min_[0], pre.scalers[1][1].data_min_[0]]
            [_, My] = [pre.scalers[0][1].data_max_[0], pre.scalers[1][1].data_max_[0]]
            return np.where(nan_in_range, np.array([mx, My]), sample_range)
        except ValueError:
            # Empty persistence diagram case - https://github.com/GUDHI/gudhi-devel/issues/507
            pass
    return sample_range


class Entropy(BaseEstimator, TransformerMixin):
    """
    This is a class for computing persistence entropy. Persistence entropy is a statistic for persistence diagrams inspired from Shannon entropy. This statistic can also be used to compute a feature vector, called the entropy summary function. See https://arxiv.org/pdf/1803.08304.pdf for more details. Note that a previous implementation was contributed by Manuel Soriano-Trigueros.
    """

    def __init__(
        self, mode="scalar", normalized=True, resolution=100, sample_range=[np.nan, np.nan]
    ):
        """
        Constructor for the Entropy class.

        Parameters:
            mode (string): what entropy to compute: either "scalar" for computing the entropy statistics, or "vector" for computing the entropy summary functions (default "scalar").
            normalized (bool): whether to normalize the entropy summary function (default True). Used only if **mode** = "vector".
            resolution (int): number of sample for the entropy summary function (default 100). Used only if **mode** = "vector".
            sample_range ([double, double]): minimum and maximum of the entropy summary function domain, of the form [x_min, x_max] (default [numpy.nan, numpy.nan]). It is the interval on which samples will be drawn evenly. If one of the values is numpy.nan, it can be computed from the persistence diagrams with the fit() method. Used only if **mode** = "vector".
        """
        self.mode, self.normalized, self.resolution, self.sample_range = (
            mode,
            normalized,
            resolution,
            sample_range,
        )

    def fit(self, X, y=None):
        """
        Fit the Entropy class on a list of persistence diagrams.

        Parameters:
            X (list of n x 2 numpy arrays): input persistence diagrams.
            y (n x 1 array): persistence diagram labels (unused).
        """
        self.sample_range = _automatic_sample_range(np.array(self.sample_range), X, y)
        return self

    def transform(self, X):
        """
        Compute the entropy for each persistence diagram individually and concatenate the results.

        Parameters:
            X (list of n x 2 numpy arrays): input persistence diagrams.

        Returns:
            numpy array with shape (number of diagrams) x (1 if **mode** = "scalar" else **resolution**): output entropy.
        """
        num_diag, Xfit = len(X), []
        x_values = np.linspace(self.sample_range[0], self.sample_range[1], self.resolution)
        step_x = x_values[1] - x_values[0]
        new_X = BirthPersistenceTransform().fit_transform(X)

        for i in range(num_diag):
            orig_diagram, diagram, num_pts_in_diag = X[i], new_X[i], X[i].shape[0]
            try:
                # new_diagram = DiagramScaler(use=True, scalers=[([1], MaxAbsScaler())]).fit_transform([diagram])[0]
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
            log0 = lambda x: 0 if x == 0 else np.log(x)
            log0 = np.frompyfunc(log0, 1, 1)
            if self.mode == "scalar":
                ent = -np.dot(p, log0(p))
                Xfit.append(np.array([[ent]]))
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
                        ent[min_idx:max_idx] -= p[j] * log0(p[j])
                if self.normalized:
                    ent = ent / np.linalg.norm(ent, ord=1)
                Xfit.append(np.reshape(ent, [1, -1]))

        Xfit = np.concatenate(Xfit, axis=0)
        return Xfit

    def __call__(self, diag):
        """
        Apply Entropy on a single persistence diagram and outputs the result.

        Parameters:
            diag (n x 2 numpy array): input persistence diagram.

        Returns:
            numpy array with shape (1 if **mode** = "scalar" else **resolution**): output entropy.
        """
        return self.fit_transform([diag])[0, :]


# Average of Birth and Death of the barcode
def _births(bar: Barcode[M]) -> NumpyVector[M, np.float64]:
    return bar[:, 0]


def _deaths(bar: Barcode[M]) -> NumpyVector[M, np.float64]:
    return bar[:, 1]


def _midpts(bar: Barcode[M]) -> NumpyVector[M, np.float64]:
    return (bar[:, 0] + bar[:, 1]) / 2


def _lengths(bar: Barcode[M]) -> NumpyVector[M, np.float64]:
    return np.abs(bar[:, 1] - bar[:, 0])


def _iqr(x: NumpyVector[int, np.floating]) -> np.floating:
    return np.subtract(*np.percentile(x, [75, 25]))


def _ran(x: NumpyVector[int, np.floating]) -> np.floating:
    return np.ptp(x)


def _p10(x: NumpyVector[int, np.floating]) -> np.floating:
    return np.percentile(x, 10, axis=None)


def _p25(x: NumpyVector[int, np.floating]) -> np.floating:
    return np.percentile(x, 25, axis=None)


def _p75(x: NumpyVector[int, np.floating]) -> np.floating:
    return np.percentile(x, 75, axis=None)


def _p90(x: NumpyVector[int, np.floating]) -> np.floating:
    return np.percentile(x, 90, axis=None)


_var_funcs = {
    "birth": _births,
    "death": _deaths,
    "midpt": _midpts,
    "length": _lengths,
}
_statistic_funcs = {
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


def get_pers_stats(barcode: Barcode[M]) -> np.ndarray[tuple[Literal[38]], np.dtype[np.float64]]:
    finite_bars = barcode[~np.any(~np.isfinite(barcode), axis=1)]
    finite_bars = _bar_cleaner(finite_bars)
    assert len(finite_bars.shape) == 2 and finite_bars.shape[1] == 2, "Got shape: " + str(
        finite_bars.shape,
    )
    if np.size(finite_bars) > 0:
        stats = np.array(
            [
                statistic_func(var_func(finite_bars))
                for var_func in _var_funcs.values()
                for statistic_func in _statistic_funcs.values()
            ]
            + [len(barcode), Entropy()(finite_bars).item()],
            dtype=np.float64,
        )
    else:
        stats = np.array([np.nan] * (len(_var_funcs) * len(_statistic_funcs) + 2), dtype=np.float64)

    stats[~np.isfinite(stats)] = 0

    return stats


pers_stats_names = [
    statistic_name + "_" + var_name
    for var_name in _var_funcs
    for statistic_name in _statistic_funcs
] + ["num_bars", "entropy"]

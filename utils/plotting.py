import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from scipy.stats import gaussian_kde
from typing import Union

__all__ = ["weighted_violinplot"]


def weighted_violinplot(
	dataset: pd.DataFrame,
	value: str,  # name of variable to plot
	*,
	group_by: str | None = None,  # name of variable to group by
	weight: str | None = None,  # name of variable indicating weights
	size: str | None = None,  # name of variable scaling the point sizes
	colour: str | None = None,  # name of variable to colour the points by
	ax: Axes | None = None,
	violin_kwargs: dict | None = None,
	scatter_kwargs: dict | None = None,
) -> tuple[Axes, dict, list[PathCollection] | PathCollection]:
	if scatter_kwargs is None:
		scatter_kwargs = {}
	if violin_kwargs is None:
		violin_kwargs = {}
	vpstats = []
	dataset = dataset.copy()
	if group_by is None:
		dataset["group_by"] = np.ones(dataset.shape[0])
		group_by = "group_by"
	if weight is None:
		dataset["weight"] = np.ones(dataset.shape[0])
		weight = "weight"
	groups = dataset[group_by].unique()
	for group in groups:
		data = dataset.loc[dataset[group_by] == group]
		kde = gaussian_kde(data[value], weights=data[weight])
		coords = np.linspace(data[value].min(), data[value].max(), 1000)
		vals = kde.evaluate(coords)
		mean = data[value].mean()
		median = data[value].median()
		min_ = data[value].min()
		max_ = data[value].max()
		vpstats.append(
			{
				"coords": coords,
				"vals": vals,
				"mean": mean,
				"median": median,
				"min": min_,
				"max": max_,
			},
		)
	ax = plt.gca() if ax is None else ax
	vplot = ax.violin(
		vpstats,
		showmeans=True,
		showmedians=True,
		**violin_kwargs,
	)
	vplot["cmedians"].set(linestyle="-")
	vplot["cmeans"].set(linestyle="--")
	plt.legend([vplot["cmedians"], vplot["cmeans"]], ["median", "mean"])
	rng = np.random.default_rng()
	sc = []
	for i, group in enumerate(groups):
		data = dataset.loc[dataset[group_by] == group]
		s = None if size is None else 100 * data[size] / data[size].max()
		c = None if colour is None else data[colour]
		sc.append(ax.scatter(
			rng.normal([i + 1] * data[value].shape[0], 0.02),
			data[value],
			s=s,
			alpha=0.7,
			c=c,
			**scatter_kwargs,
		))
	sc = sc[0] if len(sc) == 1 else sc
	return ax, vplot, sc


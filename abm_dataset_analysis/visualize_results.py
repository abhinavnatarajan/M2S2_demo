import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", auto_download=["ipynb", "html"])

with app.setup:
	import json

	import marimo as mo
	import numpy as np
	import plotly.graph_objects as go
	import polars as pl
	from plotly.colors import sample_colorscale
	from scipy.optimize import linear_sum_assignment


@app.cell(hide_code=True)
def _():
	mo.md("""
    # ABM clustering results

    Compare the EEE annotations with the saved KMeans partition, inspect agreement
    within each ABM parameter pair, and explore grouped permutation importances.
    This notebook reads the revised pipeline output without refitting the model.
    All figures use Plotly and are displayed inline.
    """)
	return


@app.cell
def load_results():
	results_path = mo.notebook_location() / "results" / "eee_clustering_results_26Oct2026.json"
	with results_path.open() as _file:
		clustering_results = json.load(_file)

	results_df = pl.DataFrame(clustering_results["results"]).sort("chi_c^m", "c_half")
	regime_order = sorted(results_df["label"].unique().to_list())
	regime_codes = {regime: index for index, regime in enumerate(regime_order)}
	regime_colors = sample_colorscale("Cividis", np.linspace(0, 1, len(regime_order)))
	plot_config = {
		"responsive": True,
		"displaylogo": False,
		"modeBarButtonsToRemove": ["toImage"],
	}
	metric_names = {
		"ari": "Adjusted Rand index",
		"nmi": "Normalized mutual information",
		"binder_loss": "Binder loss",
	}
	return (
		clustering_results,
		metric_names,
		plot_config,
		regime_codes,
		regime_colors,
		regime_order,
		results_df,
		results_path,
	)


@app.cell(hide_code=True)
def run_summary(clustering_results, results_df, results_path):
	_method = clustering_results.get("permutation_importance_method", "fixed_centroids")
	_method_label = {
		"refit_kmeans": "KMeans refitted for every permutation",
		"fixed_centroids": "Legacy fixed-centroid predictions; predates permutation refits",
	}.get(_method, _method)
	_summary_lines = [
		f"**Source:** `{results_path.name}`",
		"",
		f"**{clustering_results['num_samples']:,} simulations** · "
		f"**{clustering_results['num_features']:,} features** · "
		f"**{clustering_results['k']} clusters** · **{len(results_df)} parameter pairs** · "
		f"standardized: **{clustering_results['standardize']}** · no PCA",
		"",
		f"**Permutation method in this file:** {_method_label}.",
		"",
		"| Metric | Saved score | Direction |",
		"| :-- | --: | :-- |",
		f"| Adjusted Rand index | {clustering_results['final_model_scores']['ari']:.4f} | Higher is better |",
		f"| Normalized mutual information | {clustering_results['final_model_scores']['nmi']:.4f} | Higher is better |",
		f"| Binder loss | {clustering_results['final_model_scores']['binder_loss']:.4f} | Lower is better |",
		"",
		"These scores compare individual simulations with their EEE annotations. "
		"The parameter maps summarize the dominant cluster within each parameter pair.",
	]
	mo.md("\n".join(_summary_lines))
	return


@app.cell
def align_clusters(clustering_results, regime_order, results_df):
	# Recover each saved simulation's parameter pair so alignment uses all samples.
	_simulation_params = pl.read_csv(
		mo.notebook_location() / "data" / "simulation_params.csv",
		columns=[
			"ID",
			"chi_macrophageToCSF",
			"halfMaximalExtravasationCsf1Conc",
		],
	).rename(
		{
			"chi_macrophageToCSF": "chi_c^m",
			"halfMaximalExtravasationCsf1Conc": "c_half",
		}
	)
	samples_df = (
		pl.DataFrame(
			{
				"ID": clustering_results["sample_ids"],
				"cluster": clustering_results["labels"],
			}
		)
		.join(_simulation_params, on="ID", how="left", validate="1:1")
		.join(
			results_df.select("c_half", "chi_c^m", "label"),
			on=["c_half", "chi_c^m"],
			how="left",
			validate="m:1",
		)
	)
	if (
		samples_df.height != clustering_results["num_samples"]
		or samples_df.null_count().to_numpy().any()
	):
		raise ValueError("Saved simulation IDs must match the parameter table and EEE annotations.")

	_cluster_ids = sorted(samples_df["cluster"].unique().to_list())
	_contingency = np.array(
		[
			[
				samples_df.filter(
					(pl.col("label") == regime) & (pl.col("cluster") == cluster)
				).height
				for cluster in _cluster_ids
			]
			for regime in regime_order
		]
	)
	# Cluster numbers are arbitrary; maximize matched samples for display colors.
	_regime_indices, _cluster_indices = linear_sum_assignment(_contingency, maximize=True)
	cluster_to_regime = {
		_cluster_ids[col]: regime_order[row]
		for row, col in zip(_regime_indices, _cluster_indices, strict=True)
	}
	parameter_results_df = results_df.with_columns(
		pl.col("predicted_class").replace_strict(cluster_to_regime).alias("predicted_regime")
	).join(
		samples_df.group_by("c_half", "chi_c^m").len(name="num_samples"),
		on=["c_half", "chi_c^m"],
		how="left",
		validate="1:1",
	)
	cluster_alignment_df = pl.DataFrame(
		[
			{
				"cluster": cluster,
				"display_regime": cluster_to_regime[cluster],
				"num_simulations": samples_df.filter(pl.col("cluster") == cluster).height,
			}
			for cluster in _cluster_ids
		]
	)
	cluster_contingency_df = pl.DataFrame(
		{
			"EEE regime": regime_order,
			**{
				f"Cluster {cluster}": _contingency[:, index]
				for index, cluster in enumerate(_cluster_ids)
			},
		}
	)
	return cluster_alignment_df, cluster_contingency_df, parameter_results_df


@app.function(hide_code=True)
def parameter_grid(frame, value):
	"""Keep both ABM parameters in numeric order, without averaging category codes."""
	xvalues = frame["c_half"].unique().sort().to_list()
	grid = frame.pivot(on="c_half", index="chi_c^m", values=value).sort("chi_c^m")
	return (
		xvalues,
		grid["chi_c^m"].to_list(),
		grid.select([str(x) for x in xvalues]).to_numpy(),
	)


@app.cell(hide_code=True)
def parameter_heatmap(regime_codes, regime_colors, regime_order):
	def parameter_heatmap(frame, value, *, title, categorical=False):
		"""Render a parameter map with square cells and increasing chi along the y-axis."""
		xvalues, yvalues, matrix = parameter_grid(frame, value)
		hover_data = [
			[[x, y, matrix[row, col]] for col, x in enumerate(xvalues)]
			for row, y in enumerate(yvalues)
		]
		if categorical:
			z = [[regime_codes[label] for label in row] for row in matrix]
			colorscale = [
				[boundary, color]
				for index, color in enumerate(regime_colors)
				for boundary in (
					index / len(regime_order),
					(index + 1) / len(regime_order),
				)
			]
			heatmap = go.Heatmap(
				z=z,
				customdata=hover_data,
				zmin=-0.5,
				zmax=len(regime_order) - 0.5,
				colorscale=colorscale,
				showscale=False,
				xgap=1,
				ygap=1,
				hovertemplate="c_half: %{customdata[0]}<br>χ_cm: %{customdata[1]}<br>%{customdata[2]}<extra></extra>",
			)
		else:
			heatmap = go.Heatmap(
				z=matrix,
				customdata=hover_data,
				zmin=0,
				zmax=1,
				colorscale="Cividis",
				xgap=1,
				ygap=1,
				colorbar={"title": "Purity", "thickness": 15},
				hovertemplate="c_half: %{customdata[0]}<br>χ_cm: %{customdata[1]}<br>Purity: %{z:.1%}<extra></extra>",
			)
		figure = go.Figure(heatmap)
		if categorical:
			for regime, color in zip(regime_order, regime_colors, strict=True):
				figure.add_trace(
					go.Scatter(
						x=[None],
						y=[None],
						mode="markers",
						name=regime,
						marker={"symbol": "square", "size": 13, "color": color},
						hoverinfo="skip",
					)
				)
		figure.update_layout(
			template="plotly_white",
			title=title,
			height=620,
			font={"family": "Helvetica, Arial, sans-serif", "size": 14},
			margin={"l": 75, "r": 155, "t": 75, "b": 65},
			plot_bgcolor="#202020",
			legend={"title": "Behaviour", "x": 1.02, "y": 1},
		)
		figure.update_xaxes(
			title="c₁/₂",
			tickmode="array",
			tickvals=list(range(len(xvalues))),
			ticktext=[f"{x:.1f}" for x in xvalues],
			range=[-0.5, len(xvalues) - 0.5],
			showgrid=False,
			zeroline=False,
			constrain="domain",
		)
		figure.update_yaxes(
			title="χ_cᵐ",
			tickmode="array",
			tickvals=list(range(len(yvalues))),
			ticktext=[f"{y:.1f}" for y in yvalues],
			range=[-0.5, len(yvalues) - 0.5],
			showgrid=False,
			zeroline=False,
			scaleanchor="x",
			scaleratio=1,
			constrain="domain",
		)
		return figure

	return (parameter_heatmap,)


@app.cell(hide_code=True)
def maps_heading():
	mo.md("""
    ## Behaviour across the parameter sweep

    The three maps share the same parameter ordering. Predicted cluster numbers are
    aligned to regime names only for the display colors; this does not change the
    partition or the saved scores. Purity is the fraction of simulations assigned to
    the dominant cluster within a parameter pair, irrespective of its EEE annotation.
    """)
	return


@app.cell(hide_code=True)
def subjective_map(parameter_heatmap, parameter_results_df, plot_config):
	subjective_labels_figure = parameter_heatmap(
		parameter_results_df,
		"label",
		title="Subjective EEE annotations",
		categorical=True,
	)
	mo.ui.plotly(subjective_labels_figure, config=plot_config)
	return


@app.cell(hide_code=True)
def predicted_map(
	clustering_results,
	parameter_heatmap,
	parameter_results_df,
	plot_config,
):
	predicted_labels_figure = parameter_heatmap(
		parameter_results_df,
		"predicted_regime",
		title=f"Dominant predicted regime · k={clustering_results['k']}",
		categorical=True,
	)
	mo.ui.plotly(predicted_labels_figure, config=plot_config)
	return


@app.cell(hide_code=True)
def purity_map(parameter_heatmap, parameter_results_df, plot_config):
	purity_figure = parameter_heatmap(
		parameter_results_df,
		"purity",
		title="Within-parameter cluster purity",
	)
	mo.ui.plotly(purity_figure, config=plot_config)
	return


@app.cell(hide_code=True)
def alignment_details(
	cluster_alignment_df,
	cluster_contingency_df,
	parameter_results_df,
):
	mo.accordion(
		{
			"Cluster alignment and result tables": mo.vstack(
				[
					mo.md(
						"Alignment maximizes the number of matched **individual simulations**. "
						"Rows below are EEE annotations; columns are the original cluster numbers."
					),
					mo.ui.table(
						cluster_contingency_df,
						selection=None,
						show_download=False,
					),
					mo.ui.table(
						cluster_alignment_df,
						selection=None,
						show_download=False,
					),
					mo.ui.table(
						parameter_results_df,
						selection=None,
						page_size=10,
						show_download=False,
					),
				]
			)
		}
	)
	return


@app.cell
def importance_summary(clustering_results):
	# Match the colorectal charts: compute sample SD across permutations (ddof=1).
	_importance_rows = []
	for _group in clustering_results["feature_importances_for_ground_truth"]:
		_cell_group = "/".join(_group["cell_group"])
		_group_name = f"{_cell_group} · {_group['diagram']}{_group['dimension']}"
		for _metric, _raw_values in _group["importances"].items():
			_values = np.asarray(_raw_values, dtype=float)
			_mean = float(_values.mean())
			_sd = float(_values.std(ddof=1)) if len(_values) > 1 else 0.0
			_importance_rows.append(
				{
					"feature_group": _group_name,
					"cell_group": _cell_group,
					"diagram": _group["diagram"],
					"dimension": _group["dimension"],
					"num_columns": len(_group["columns"]),
					"metric": _metric,
					"num_permutations": len(_values),
					"average_importance": _mean,
					"median_importance": float(np.median(_values)),
					"standard_deviation": _sd,
					"minimum_importance": float(_values.min()),
					"maximum_importance": float(_values.max()),
					"box_lower": _mean - 2 * _sd,
					"box_upper": _mean + 2 * _sd,
				}
			)
	feature_importance_summary_df = pl.DataFrame(_importance_rows)
	all_importances_zero = (
		feature_importance_summary_df["minimum_importance"].eq(0).all()
		and feature_importance_summary_df["maximum_importance"].eq(0).all()
	)
	return all_importances_zero, feature_importance_summary_df


@app.cell(hide_code=True)
def importance_heading(all_importances_zero, clustering_results):
	_refitted = clustering_results.get("permutation_importance_method") == "refit_kmeans"
	_method_description = (
		"Each permuted dataset is fitted with a fresh copy of the KMeans pipeline, "
		"including imputation and scaling, with the same settings and clustering seed."
		if _refitted
		else "**This saved file uses the legacy fixed-centroid method.** The revised pipeline "
		"refits KMeans for each permutation; generate a new results file to view those importances."
	)
	_description = "\n\n".join(
		[
			"## Grouped permutation importance",
			"Each group contains the statistics for one cell group, diagram, and homology degree. "
			f"There are **{clustering_results['num_permutations']} permutations** per group.",
			_method_description,
			"ARI/NMI importance is baseline minus permuted score; Binder importance is permuted "
			"minus baseline loss. Positive importance means worse agreement after permutation "
			"for every metric; negative values are retained. EEE annotations enter scoring only.",
			"As in the colorectal notebook: **box = mean ± 2 SD**, solid line = median, "
			"dashed line = mean, whiskers = full minimum–maximum range. SD uses `ddof=1`. "
			"The scores describe the observed simulations and are not held-out performance.",
		]
	)
	mo.vstack(
		[
			mo.md(_description),
			mo.callout(
				mo.md(
					f"**All {len(clustering_results['feature_importances_for_ground_truth'])} feature groups "
					"have exactly zero saved importance for all three metrics.** "
					"The recorded scores did not change under these permutations, so this file "
					"provides no ranking between the groups. Tied groups are displayed alphabetically."
				)
				if all_importances_zero
				else mo.md(
					"Groups are ordered by decreasing mean importance, with alphabetical ordering for ties."
				),
				kind="warn" if all_importances_zero else "info",
			),
		]
	)
	return


@app.cell
def importance_controls(clustering_results, metric_names):
	importance_metric = mo.ui.dropdown(
		options={label: key for key, label in metric_names.items()},
		value="Adjusted Rand index",
		allow_select_none=False,
		label="Importance metric",
	)
	importance_group_count = mo.ui.slider(
		start=1,
		stop=len(clustering_results["feature_importances_for_ground_truth"]),
		step=1,
		value=20,
		show_value=True,
		include_input=True,
		debounce=True,
		label="Groups shown",
	)
	mo.hstack([importance_metric, importance_group_count], justify="start", gap=3)
	return importance_group_count, importance_metric


@app.cell(hide_code=True)
def importance_chart(
	all_importances_zero,
	feature_importance_summary_df,
	importance_group_count,
	importance_metric,
	metric_names,
	plot_config,
):
	selected_importance_df = feature_importance_summary_df.filter(
		pl.col("metric") == importance_metric.value
	).sort(["average_importance", "feature_group"], descending=[True, False])
	displayed_importance_df = selected_importance_df.head(importance_group_count.value)
	_hover_data = displayed_importance_df.select(
		"average_importance",
		"median_importance",
		"standard_deviation",
		"box_lower",
		"box_upper",
		"minimum_importance",
		"maximum_importance",
		"num_columns",
	).rows()
	feature_importance_chart = go.Figure(
		go.Box(
			y=displayed_importance_df["feature_group"].to_list(),
			q1=displayed_importance_df["box_lower"].to_list(),
			median=displayed_importance_df["median_importance"].to_list(),
			q3=displayed_importance_df["box_upper"].to_list(),
			mean=displayed_importance_df["average_importance"].to_list(),
			sd=displayed_importance_df["standard_deviation"].to_list(),
			lowerfence=displayed_importance_df["minimum_importance"].to_list(),
			upperfence=displayed_importance_df["maximum_importance"].to_list(),
			orientation="h",
			sizemode="quartiles",
			showwhiskers=True,
			boxmean=True,
			boxpoints=False,
			fillcolor="rgba(99, 110, 250, 0.45)",
			line={"color": "#1F2937", "width": 2},
			customdata=_hover_data,
			name="",
			hovertemplate=(
				"<b>%{y}</b><br>Mean: %{customdata[0]:.6g}<br>Median: %{customdata[1]:.6g}"
				"<br>SD: %{customdata[2]:.6g}<br>Mean − 2 SD: %{customdata[3]:.6g}"
				"<br>Mean + 2 SD: %{customdata[4]:.6g}<br>Minimum: %{customdata[5]:.6g}"
				"<br>Maximum: %{customdata[6]:.6g}<br>Columns permuted together: %{customdata[7]}"
				"<extra></extra>"
			),
		)
	)
	feature_importance_chart.update_layout(
		template="plotly_white",
		title=(
			f"Permutation importance · {metric_names[importance_metric.value]}"
			"<br><sup>Box: mean ± 2 SD · solid: median · dashed: mean · whiskers: full range</sup>"
		),
		xaxis_title="Score deterioration after grouped permutation",
		yaxis_title="Cell group · diagram and homology degree",
		yaxis={
			"categoryorder": "array",
			"categoryarray": displayed_importance_df["feature_group"].reverse().to_list(),
			"tickfont": {"size": 12},
		},
		font={"family": "Helvetica, Arial, sans-serif", "size": 14},
		height=max(500, 30 * displayed_importance_df.height + 150),
		margin={"l": 370, "r": 30, "t": 90, "b": 70},
		showlegend=False,
	)
	if all_importances_zero:
		feature_importance_chart.update_xaxes(range=[-0.01, 0.01], tickformat=".3f")
	_caption = (
		f"Showing {displayed_importance_df.height} of {selected_importance_df.height} groups "
		+ (
			"in alphabetical order: all means and spreads are zero."
			if all_importances_zero
			else "in decreasing order of mean importance."
		)
	)
	mo.vstack(
		[
			mo.md(_caption),
			mo.ui.plotly(feature_importance_chart, config=plot_config),
			mo.accordion(
				{
					"All feature-importance summaries for this metric": mo.ui.table(
						selected_importance_df,
						selection=None,
						page_size=10,
						show_download=False,
						freeze_columns_left=["feature_group"],
					),
				}
			),
		]
	)
	return


if __name__ == "__main__":
	app.run()

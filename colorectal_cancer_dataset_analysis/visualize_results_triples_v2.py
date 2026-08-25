import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", auto_download=["ipynb"])

with app.setup:
    import marimo as mo
    import polars as pl
    import pyarrow.parquet as pq
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go


@app.cell
def _():
    classification_pairs_df = pl.read_parquet(
    	"./colorectal_cancer_dataset_analysis/results/classification_triples_v2/**/*.parquet",
    	hive_partitioning=True,
    ).select(
    	"patient_id",
    	"num_samples",
    	"cv_oof_score",
    	pl.exclude("patient_id", "num_samples", "cv_oof_score"),
    )
    classification_pairs_df
    return (classification_pairs_df,)


@app.cell
def _(classification_pairs_df):
    patient_score_summary_df = (
    	classification_pairs_df.with_columns(
    		pl.col("cv_test_scores").list.mean().alias("cv_test_scores_mean"),
    		pl.col("cv_test_scores").list.std().alias("cv_test_scores_std"),
    		pl.struct("cv_train_scores", "cv_test_scores")
    		.map_elements(
    			lambda row: [
    				train - test
    				for train, test in zip(
    					row["cv_train_scores"], row["cv_test_scores"]
    				)
    			],
    			return_dtype=pl.List(pl.Float64),
    		)
    		.alias("_cv_train_test_differences"),
    	)
    	.with_columns(
    		pl.col("_cv_train_test_differences")
    		.list.mean()
    		.alias("cv_train_test_difference_mean"),
    		pl.col("_cv_train_test_differences")
    		.list.std()
    		.alias("cv_train_test_difference_std"),
    	)
    	.select(
    		"patient_id",
    		"num_samples",
    		"cv_oof_score",
    		"cv_test_scores_mean",
    		"cv_test_scores_std",
    		"cv_train_test_difference_mean",
    		"cv_train_test_difference_std",
    		pl.exclude(
    			"patient_id",
    			"num_samples",
    			"cv_oof_score",
    			"cv_test_scores_mean",
    			"cv_test_scores_std",
    			"cv_train_test_difference_mean",
    			"cv_train_test_difference_std",
    			"_cv_train_test_differences",
    		),
    	)
    )
    patient_score_summary_df
    return


@app.cell
def _(classification_pairs_df):
    _patient_ids = classification_pairs_df["patient_id"].sort().to_list()
    patient_id_dropdown = mo.ui.dropdown(
    	options={str(patient_id): patient_id for patient_id in _patient_ids},
    	value=str(_patient_ids[0]),
    	label="Patient ID",
    	searchable=True,
    )
    return (patient_id_dropdown,)


@app.cell
def _(classification_pairs_df, patient_id_dropdown):
    _selected_feature_importances = (
    	classification_pairs_df.filter(
    		pl.col("patient_id") == patient_id_dropdown.value
    	)
    	.select("feature_importances")
    	.item()
    )
    _feature_importance_data = (
    	pl.DataFrame(
    		{
    			"cell_group": list(_selected_feature_importances),
    			"average_importance": [
    				float(np.mean(values))
    				for values in _selected_feature_importances.values()
    			],
    			"median_importance": [
    				float(np.median(values))
    				for values in _selected_feature_importances.values()
    			],
    			"standard_deviation": [
    				float(np.std(values, ddof=1))
    				for values in _selected_feature_importances.values()
    			],
    			"minimum_importance": [
    				float(np.min(values))
    				for values in _selected_feature_importances.values()
    			],
    			"maximum_importance": [
    				float(np.max(values))
    				for values in _selected_feature_importances.values()
    			],
    		}
    	)
    	.with_columns(
    		(
    			pl.col("average_importance") - 2 * pl.col("standard_deviation")
    		).alias("box_lower"),
    		(
    			pl.col("average_importance") + 2 * pl.col("standard_deviation")
    		).alias("box_upper"),
    	)
    	.sort("average_importance", descending=True)
    	.head(20)
    )
    _feature_importance_hover_data = _feature_importance_data.select(
    	"average_importance",
    	"median_importance",
    	"standard_deviation",
    	"box_lower",
    	"box_upper",
    	"minimum_importance",
    	"maximum_importance",
    ).rows()
    feature_importance_chart = go.Figure(
    	go.Box(
    		y=_feature_importance_data["cell_group"].to_list(),
    		q1=_feature_importance_data["box_lower"].to_list(),
    		median=_feature_importance_data["median_importance"].to_list(),
    		q3=_feature_importance_data["box_upper"].to_list(),
    		mean=_feature_importance_data["average_importance"].to_list(),
    		sd=_feature_importance_data["standard_deviation"].to_list(),
    		lowerfence=_feature_importance_data["minimum_importance"].to_list(),
    		upperfence=_feature_importance_data["maximum_importance"].to_list(),
    		orientation="h",
    		sizemode="quartiles",
    		showwhiskers=True,
    		boxmean=True,
    		boxpoints=False,
    		fillcolor="rgba(99, 110, 250, 0.45)",
    		line={"color": "#1F2937", "width": 2},
    		customdata=_feature_importance_hover_data,
    		hovertemplate=(
    			"<b>%{y}</b><br>"
    			"Mean: %{customdata[0]:.6g}<br>"
    			"Median: %{customdata[1]:.6g}<br>"
    			"SD: %{customdata[2]:.6g}<br>"
    			"Mean - 2 SD: %{customdata[3]:.6g}<br>"
    			"Mean + 2 SD: %{customdata[4]:.6g}<br>"
    			"Minimum: %{customdata[5]:.6g}<br>"
    			"Maximum: %{customdata[6]:.6g}"
    			"<extra></extra>"
    		),
    		name="",
    	)
    )
    feature_importance_chart.update_layout(
    	title=(
    		f"Top feature importances for patient {patient_id_dropdown.value}"
    		"<br><sup>Box: mean +/- 2 SD; dashed line: mean; solid line: median; "
    		"whiskers: full range</sup>"
    	),
    	xaxis_title="Feature importance",
    	yaxis_title="Cell group",
    	yaxis={
    		"categoryorder": "array",
    		"categoryarray": _feature_importance_data["cell_group"]
    		.reverse()
    		.to_list(),
    	},
    	height=700,
    	showlegend=False,
    )
    mo.vstack([patient_id_dropdown, feature_importance_chart])
    return


@app.cell
def _(classification_pairs_df):
    _feature_importance_rows = list(
    	classification_pairs_df.select("patient_id", "feature_importances")
    	.sort("patient_id")
    	.iter_rows(named=True)
    )
    _cell_groups = list(_feature_importance_rows[0]["feature_importances"])
    cell_group_importance_by_patient_df = pl.DataFrame(
    	{
    		"cell_group": _cell_groups,
    		"avg_importance_by_patient": pl.Series(
    			"avg_importance_by_patient",
    			[
    				{
    					row["patient_id"]: float(
    						np.mean(row["feature_importances"][cell_group])
    					)
    					for row in _feature_importance_rows
    				}
    				for cell_group in _cell_groups
    			],
    			dtype=pl.Object,
    		),
    	}
    )
    cell_group_importance_by_patient_df
    return (cell_group_importance_by_patient_df,)


@app.cell
def _():
    reweight_by_num_samples_checkbox = mo.ui.checkbox(
    	value=False,
    	label="reweight by number of samples",
    )
    return (reweight_by_num_samples_checkbox,)


@app.cell
def _(
    cell_group_importance_by_patient_df,
    classification_pairs_df,
    reweight_by_num_samples_checkbox,
):
    _num_samples_by_patient = dict(
    	classification_pairs_df.select("patient_id", "num_samples").iter_rows()
    )
    _patient_importance_mappings = list(
    	cell_group_importance_by_patient_df["avg_importance_by_patient"]
    )
    _patient_importance_values = [
    	list(patient_importances.values())
    	for patient_importances in _patient_importance_mappings
    ]

    def _average_importance(patient_importances):
    	if reweight_by_num_samples_checkbox.value:
    		return float(
    			sum(
    				importance * _num_samples_by_patient[patient_id]
    				for patient_id, importance in patient_importances.items()
    			)
    			/ sum(
    				_num_samples_by_patient[patient_id]
    				for patient_id in patient_importances
    			)
    		)
    	return float(np.mean(list(patient_importances.values())))

    _average_axis_label = (
    	"Sample-weighted average feature importance across patients"
    	if reweight_by_num_samples_checkbox.value
    	else "Average feature importance across patients"
    )
    _weighting_title = (
    	"sample-weighted feature importances across patients"
    	if reweight_by_num_samples_checkbox.value
    	else "feature importances across patients"
    )
    _feature_importance_across_patients_data = (
    	pl.DataFrame(
    		{
    			"cell_group": cell_group_importance_by_patient_df["cell_group"],
    			"average_importance": [
    				_average_importance(patient_importances)
    				for patient_importances in _patient_importance_mappings
    			],
    			"median_importance": [
    				float(np.median(values))
    				for values in _patient_importance_values
    			],
    			"standard_deviation": [
    				float(np.std(values, ddof=1))
    				for values in _patient_importance_values
    			],
    			"minimum_importance": [
    				float(np.min(values))
    				for values in _patient_importance_values
    			],
    			"maximum_importance": [
    				float(np.max(values))
    				for values in _patient_importance_values
    			],
    		}
    	)
    	.with_columns(
    		(
    			pl.col("average_importance") - 2 * pl.col("standard_deviation")
    		).alias("box_lower"),
    		(
    			pl.col("average_importance") + 2 * pl.col("standard_deviation")
    		).alias("box_upper"),
    	)
    	.sort("average_importance", descending=True)
    	.head(20)
    )
    _feature_importance_across_patients_hover_data = (
    	_feature_importance_across_patients_data.select(
    		"average_importance",
    		"median_importance",
    		"standard_deviation",
    		"box_lower",
    		"box_upper",
    		"minimum_importance",
    		"maximum_importance",
    	).rows()
    )
    feature_importance_across_patients_chart = go.Figure(
    	go.Box(
    		y=_feature_importance_across_patients_data["cell_group"].to_list(),
    		q1=_feature_importance_across_patients_data["box_lower"].to_list(),
    		median=_feature_importance_across_patients_data[
    			"median_importance"
    		].to_list(),
    		q3=_feature_importance_across_patients_data["box_upper"].to_list(),
    		mean=_feature_importance_across_patients_data[
    			"average_importance"
    		].to_list(),
    		sd=_feature_importance_across_patients_data[
    			"standard_deviation"
    		].to_list(),
    		lowerfence=_feature_importance_across_patients_data[
    			"minimum_importance"
    		].to_list(),
    		upperfence=_feature_importance_across_patients_data[
    			"maximum_importance"
    		].to_list(),
    		orientation="h",
    		sizemode="quartiles",
    		showwhiskers=True,
    		boxmean=True,
    		boxpoints=False,
    		fillcolor="rgba(99, 110, 250, 0.45)",
    		line={"color": "#1F2937", "width": 2},
    		customdata=_feature_importance_across_patients_hover_data,
    		hovertemplate=(
    			"<b>%{y}</b><br>"
    			"Mean: %{customdata[0]:.6g}<br>"
    			"Median: %{customdata[1]:.6g}<br>"
    			"SD: %{customdata[2]:.6g}<br>"
    			"Mean - 2 SD: %{customdata[3]:.6g}<br>"
    			"Mean + 2 SD: %{customdata[4]:.6g}<br>"
    			"Minimum: %{customdata[5]:.6g}<br>"
    			"Maximum: %{customdata[6]:.6g}"
    			"<extra></extra>"
    		),
    		name="",
    	)
    )
    feature_importance_across_patients_chart.update_layout(
    	title=(
    		f"Top {_weighting_title}"
    		"<br><sup>Box: mean +/- 2 SD; dashed line: mean; solid line: median; "
    		"whiskers: full range</sup>"
    	),
    	xaxis_title=_average_axis_label,
    	yaxis_title="Cell group",
    	yaxis={
    		"categoryorder": "array",
    		"categoryarray": _feature_importance_across_patients_data[
    			"cell_group"
    		]
    		.reverse()
    		.to_list(),
    	},
    	height=700,
    	showlegend=False,
    )
    mo.vstack(
    	[
    		reweight_by_num_samples_checkbox,
    		feature_importance_across_patients_chart,
    	]
    )
    return


@app.cell
def _(cell_group_importance_by_patient_df, classification_pairs_df):
    from sklearn.decomposition import PCA as _PCA

    _pca_cell_groups = cell_group_importance_by_patient_df[
    	"cell_group"
    ].to_list()
    _pca_patient_ids = sorted(
    	classification_pairs_df["patient_id"].unique().to_list()
    )
    _pca_importance_mappings = cell_group_importance_by_patient_df[
    	"avg_importance_by_patient"
    ].to_list()
    patient_feature_importance_vectors_df = pl.DataFrame(
    	{
    		"patient_id": _pca_patient_ids,
    		**{
    			_cell_group: [
    				float(_importance_mapping[_patient_id])
    				for _patient_id in _pca_patient_ids
    			]
    			for _cell_group, _importance_mapping in zip(
    				_pca_cell_groups,
    				_pca_importance_mappings,
    			)
    		},
    	}
    )

    _patient_importance_matrix = patient_feature_importance_vectors_df.select(
    	_pca_cell_groups
    ).to_numpy()
    _pca_model = _PCA(n_components=3)
    _patient_pca_scores = _pca_model.fit_transform(_patient_importance_matrix)
    _pca_component_names = ["PCA1", "PCA2", "PCA3"]
    _pca_explained_variance_percent = 100 * _pca_model.explained_variance_ratio_

    patient_pca_scores_df = pl.DataFrame(
    	{
    		"patient_id": _pca_patient_ids,
    		**{
    			_component_name: _patient_pca_scores[:, _component_index]
    			for _component_index, _component_name in enumerate(
    				_pca_component_names
    			)
    		},
    	}
    )
    _pca_scatter_data = patient_pca_scores_df.to_dict(as_series=False)
    patient_pca_scatter = px.scatter_3d(
    	_pca_scatter_data,
    	x="PCA1",
    	y="PCA2",
    	z="PCA3",
    	hover_name="patient_id",
    	labels={
    		_component_name: (
    			f"{_component_name} "
    			f"({_pca_explained_variance_percent[_component_index]:.1f}% variance)"
    		)
    		for _component_index, _component_name in enumerate(
    			_pca_component_names
    		)
    	},
    	title="Patients in feature-importance PCA space",
    )
    patient_pca_scatter.update_traces(
    	marker={"size": 6, "opacity": 0.8},
    	hovertemplate=(
    		"<b>Patient %{hovertext}</b><br>"
    		"PCA1: %{x:.4g}<br>"
    		"PCA2: %{y:.4g}<br>"
    		"PCA3: %{z:.4g}<extra></extra>"
    	),
    )
    patient_pca_scatter.update_layout(height=700)

    pca_component_loadings_df = pl.DataFrame(
    	{
    		"cell_group": _pca_cell_groups * 3,
    		"component": [
    			_component_name
    			for _component_name in _pca_component_names
    			for _ in _pca_cell_groups
    		],
    		"loading": _pca_model.components_.reshape(-1),
    	}
    )

    def _make_pca_loading_chart(_component_index):
    	_component_name = _pca_component_names[_component_index]
    	_component_loadings = _pca_model.components_[_component_index]
    	_top_indices = np.argsort(_component_loadings)[::-1][:10]
    	_top_cell_groups = [_pca_cell_groups[_index] for _index in _top_indices]
    	_top_loadings = _component_loadings[_top_indices]
    	_chart = go.Figure(
    		go.Bar(
    			x=_top_loadings,
    			y=_top_cell_groups,
    			orientation="h",
    			marker_color="#636EFA",
    			hovertemplate=(
    				"<b>%{y}</b><br>"
    				+ _component_name
    				+ " loading: %{x:.5f}<extra></extra>"
    			),
    		)
    	)
    	_chart.update_layout(
    		title=(
    			f"{_component_name} loadings "
    			f"({_pca_explained_variance_percent[_component_index]:.1f}% variance explained)"
    			"<br><sup>Top 10 cell groups by loading, highest first</sup>"
    		),
    		xaxis_title="PCA loading coefficient",
    		yaxis_title="Cell group",
    		height=500,
    		showlegend=False,
    		margin={"l": 260},
    		yaxis={
    			"categoryorder": "array",
    			"categoryarray": _top_cell_groups,
    			"autorange": "reversed",
    		},
    	)
    	return _chart

    pca1_loadings_chart = _make_pca_loading_chart(0)
    pca2_loadings_chart = _make_pca_loading_chart(1)
    pca3_loadings_chart = _make_pca_loading_chart(2)

    mo.vstack(
    	[
    		mo.md(
    			f"**PCA input:** {len(_pca_patient_ids)} patients x "
    			f"{len(_pca_cell_groups)} cell groups. The first three components "
    			f"explain {_pca_explained_variance_percent.sum():.1f}% of the variance. "
    			"Component signs are arbitrary; the loading magnitudes show each "
    			"cell group's contribution."
    		),
    		patient_pca_scatter,
    		pca1_loadings_chart,
    		pca2_loadings_chart,
    		pca3_loadings_chart,
    	]
    )
    return


@app.cell
def _(cell_group_importance_by_patient_df):
    _frequency_cell_groups = cell_group_importance_by_patient_df[
    	"cell_group"
    ].to_list()
    _frequency_importance_mappings = cell_group_importance_by_patient_df[
    	"avg_importance_by_patient"
    ].to_list()
    _frequency_patient_ids = sorted(_frequency_importance_mappings[0])
    _top5_cell_group_counts = dict.fromkeys(_frequency_cell_groups, 0)

    for _patient_id in _frequency_patient_ids:
    	_ranked_cell_groups = sorted(
    		zip(_frequency_cell_groups, _frequency_importance_mappings),
    		key=lambda _item: (
    			-float(_item[1][_patient_id]),
    			_item[0],
    		),
    	)
    	for _cell_group, _ in _ranked_cell_groups[:5]:
    		_top5_cell_group_counts[_cell_group] += 1

    _top20_cell_group_frequencies = sorted(
    	_top5_cell_group_counts.items(),
    	key=lambda _item: (-_item[1], _item[0]),
    )[:20]
    top_cell_group_frequency_df = pl.DataFrame(
    	{
    		"cell_group": [
    			_cell_group for _cell_group, _ in _top20_cell_group_frequencies
    		],
    		"patient_count": [
    			_count for _, _count in _top20_cell_group_frequencies
    		],
    	}
    )
    _frequency_chart_cell_groups = top_cell_group_frequency_df[
    	"cell_group"
    ].to_list()
    _frequency_chart_patient_counts = top_cell_group_frequency_df[
    	"patient_count"
    ].to_list()
    top_cell_group_frequency_chart = go.Figure(
    	go.Bar(
    		x=_frequency_chart_patient_counts,
    		y=_frequency_chart_cell_groups,
    		orientation="h",
    		text=_frequency_chart_patient_counts,
    		textposition="outside",
    		cliponaxis=False,
    		marker_color="#636EFA",
    		hovertemplate=(
    			"<b>%{y}</b><br>"
    			"Appears in top 5 for %{x} patients<extra></extra>"
    		),
    	)
    )
    top_cell_group_frequency_chart.update_layout(
    	title=(
    		"Most frequent cell groups among each patient's top 5"
    		"<br><sup>Cell groups are ranked per patient by average feature importance</sup>"
    	),
    	xaxis_title="Number of patients",
    	yaxis_title="Cell group",
    	height=550,
    	showlegend=False,
    	margin={"l": 260, "r": 60},
    	xaxis={"dtick": 1, "rangemode": "tozero"},
    	yaxis={
    		"categoryorder": "array",
    		"categoryarray": _frequency_chart_cell_groups,
    		"autorange": "reversed",
    	},
    )

    mo.vstack(
    	[
    		mo.md(
    			f"Across {len(_frequency_patient_ids)} patients, these are the 20 "
    			"cell groups that most often appear among a patient's five highest "
    			"average feature importances."
    		),
    		top_cell_group_frequency_chart,
    	]
    )
    return


if __name__ == "__main__":
    app.run()

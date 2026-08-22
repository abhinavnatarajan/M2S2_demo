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
    	"./colorectal_cancer_dataset_analysis/results/classification_pairs_v2/**/*.parquet",
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


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", auto_download=["ipynb", "html"])


@app.cell
def overview(mo):
	mo.md(r"""
    # ABM simulation examples\n"
    	"\n"
    	"Explore the time-500 point clouds for **equilibrium**, **elimination**, and\n"
    	"**escape**, then compare representative simulations side by side.\n"
    	"\n"
    	"The analysis follows `generate_example_figures.ipynb`. Stroma is omitted from\n"
    	"all plots. Hover to inspect coordinates, zoom or pan within a panel, and click\n"
    	"a legend entry to toggle that cell type across the figure. Figures are displayed\n"
    	"inline; no figure files are written.
    """)
	return


@app.cell
def imports():
	from math import ceil

	import marimo as mo
	import plotly.graph_objects as go
	import polars as pl
	from plotly.subplots import make_subplots

	return ceil, go, make_subplots, mo, pl


@app.cell
def data(mo, pl):
	# Resolve paths relative to this notebook, independently of the kernel's cwd.
	data_dir = mo.notebook_location() / "data"
	samples_dir = data_dir / "simulation_samples"
	data_params_df = pl.read_csv(
		data_dir / "simulation_params.csv",
		columns=[
			"simulation_id",
			"ID",
			"chi_macrophageToCSF",
			"halfMaximalExtravasationCsf1Conc",
		],
	).rename(
		{
			"chi_macrophageToCSF": "chi_cm",
			"halfMaximalExtravasationCsf1Conc": "c_half",
		}
	)
	return data_params_df, samples_dir


@app.cell
def plot_styles():
	# Tableau's colorblind palette, matching the original notebook.
	colors = (
		"#006BA4",
		"#FF800E",
		"#ABABAB",
		"#595959",
		"#5F9ED1",
		"#C85200",
		"#898989",
		"#A2C8EC",
		"#FFBC79",
		"#CFCFCF",
	)
	grid_styles = {
		"Vessel": {
			"color": colors[0],
			"symbol": "circle",
			"size": 6,
			"opacity": 0.5,
		},
		"Tumour": {
			"color": colors[1],
			"symbol": "star",
			"size": 6,
			"opacity": 0.3,
		},
		"Macrophage": {
			"color": colors[2],
			"symbol": "square",
			"size": 6,
			"opacity": 0.8,
		},
		"Necrotic": {
			"color": "black",
			"symbol": "x-thin",
			"line": {"color": "black", "width": 1},
			"size": 6,
			"opacity": 0.8,
		},
	}
	regime_styles = {
		"Vessel": {
			"color": "rgba(0, 107, 164, 0.8)",
			"symbol": "circle",
			"size": 7.3,
			"line": {"color": "rgba(255, 255, 255, 0.6)", "width": 0.5},
		},
		"Tumour": {
			"color": colors[8],
			"symbol": "star",
			"size": 12,
			"line": {"color": "rgba(255, 255, 255, 0.6)", "width": 0.5},
		},
		"Macrophage": {
			"color": "rgba(200, 82, 0, 0.6)",
			"symbol": "square",
			"size": 7.3,
			"line": {"color": "rgba(255, 255, 255, 0.6)", "width": 0.5},
		},
		"Necrotic": {
			"color": "rgba(139, 0, 0, 0.8)",
			"symbol": "x",
			"size": 9.5,
			"line": {"color": "rgba(255, 255, 255, 0.6)", "width": 1},
		},
	}
	plot_config = {
		"responsive": True,
		"displaylogo": False,
		"modeBarButtonsToRemove": ["toImage"],
	}
	return grid_styles, plot_config, regime_styles


@app.cell
def sample_reader(data_params_df, pl, samples_dir):
	def sample_path(sample_id):
		"""Locate a time-500 sample in the downloaded ABM dataset."""
		return samples_dir / f"ID-{sample_id}_time-500_From2ParamSweep_Data.csv"

	def load_sample(sample_id):
		"""Read only the spatial columns, omitting stroma as in the source notebook."""
		return pl.read_csv(
			sample_path(sample_id),
			columns=["points_x", "points_y", "celltypes"],
		).filter(pl.col("celltypes") != "Stroma")

	def sample_availability(c_half, chi_cm):
		"""Describe available CSVs from the current parameter table and data folder."""
		selected = data_params_df.filter(
			(pl.col("c_half") == c_half) & (pl.col("chi_cm") == chi_cm)
		)
		missing = [
			sample_id for sample_id in selected["ID"] if not sample_path(sample_id).is_file()
		]
		summary = f"{selected.height - len(missing)} of {selected.height} samples are available."
		if missing:
			summary += " Missing CSVs are skipped: " + ", ".join(map(str, missing)) + "."
		return summary

	return load_sample, sample_availability, sample_path


@app.cell
def figure_builder(ceil, go, load_sample, make_subplots, pl):
	def make_sample_figure(samples, *, title, styles, nrows=None, hide_ticks=False):
		"""Plot (sample ID, panel title) pairs with one shared cell-type legend."""
		ncols = 3
		nrows = nrows or ceil(len(samples) / ncols)
		figure = make_subplots(
			rows=nrows,
			cols=ncols,
			subplot_titles=[label for _, label in samples],
			horizontal_spacing=0.05,
			vertical_spacing=0.035 if nrows > 1 else 0,
		)
		legend_seen = set()
		legend_rank = {name: rank for rank, name in enumerate(sorted(styles))}
		for index, (sample_id, _) in enumerate(samples):
			row, col = divmod(index, ncols)
			data = load_sample(sample_id)
			for celltype in data["celltypes"].unique(maintain_order=True):
				points = data.filter(pl.col("celltypes") == celltype)
				figure.add_trace(
					go.Scatter(
						x=points["points_x"].to_list(),
						y=points["points_y"].to_list(),
						mode="markers",
						name=celltype,
						legendgroup=celltype,
						legendrank=legend_rank[celltype],
						showlegend=celltype not in legend_seen,
						marker=styles[celltype],
						meta={"sample_id": sample_id, "celltype": celltype},
						hovertemplate=(
							"<b>%{meta.celltype}</b> · ID %{meta.sample_id}"
							"<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>"
						),
					),
					row=row + 1,
					col=col + 1,
				)
				legend_seen.add(celltype)
			# Anchor each y-axis to its own x-axis so spatial distances remain equal.
			axis_number = index + 1
			x_axis = "x" if axis_number == 1 else f"x{axis_number}"
			figure.update_yaxes(
				scaleanchor=x_axis,
				scaleratio=1,
				constrain="domain",
				row=row + 1,
				col=col + 1,
			)
			figure.update_xaxes(constrain="domain", row=row + 1, col=col + 1)
		for index in range(len(samples), nrows * ncols):
			row, col = divmod(index, ncols)
			figure.update_xaxes(visible=False, row=row + 1, col=col + 1)
			figure.update_yaxes(visible=False, row=row + 1, col=col + 1)
		figure.update_layout(
			template="plotly_white",
			title={"text": title, "x": 0.5, "xanchor": "center"},
			height=1800 if nrows == 6 else 450 if nrows == 1 else 300 * nrows,
			autosize=True,
			font={"family": "Helvetica, Arial, sans-serif", "size": 14},
			margin={"l": 35, "r": 15, "t": 95, "b": 90},
			legend={
				"orientation": "h",
				"x": 0.5,
				"xanchor": "center",
				"y": -0.04 if nrows > 1 else -0.12,
				"yanchor": "top",
				"groupclick": "togglegroup",
				"traceorder": "normal",
			},
			hovermode="closest",
		)
		figure.update_xaxes(
			showgrid=False,
			zeroline=False,
			showline=True,
			mirror=True,
			linecolor="#D0D0D0",
			showticklabels=not hide_ticks,
			ticks="" if hide_ticks else "outside",
		)
		figure.update_yaxes(
			showgrid=False,
			zeroline=False,
			showline=True,
			mirror=True,
			linecolor="#D0D0D0",
			showticklabels=not hide_ticks,
			ticks="" if hide_ticks else "outside",
		)
		return figure

	return (make_sample_figure,)


@app.cell
def parameter_grid(
	ceil,
	data_params_df,
	grid_styles,
	make_sample_figure,
	pl,
	sample_path,
):
	def show_plots(c_half, chi_cm, *, regime):
		"""Show all available samples for a parameter pair, skipping absent CSVs."""
		selected = data_params_df.filter(
			(pl.col("c_half") == c_half) & (pl.col("chi_cm") == chi_cm)
		)
		sample_ids = [sample_id for sample_id in selected["ID"] if sample_path(sample_id).is_file()]
		if not sample_ids:
			raise ValueError(f"No available samples for c_half={c_half}, chi_cm={chi_cm}.")
		return make_sample_figure(
			[(sample_id, f"ID {sample_id}") for sample_id in sample_ids],
			title=f"{regime} candidates<br><sup>c_half = {c_half} · χ_cm = {chi_cm}</sup>",
			styles=grid_styles,
			nrows=max(6, ceil(len(sample_ids) / 3)),
		)

	return (show_plots,)


@app.cell
def equilibrium_heading(mo, sample_availability):
	mo.md(
		"## Equilibrium candidates\n\n"
		"$c_{\\mathrm{half}} = 0.6$, $\\chi_{cm} = 0.5$. Representative example: **ID 112**."
		"\n\n" + sample_availability(0.6, 0.5)
	)
	return


@app.cell
def equilibrium_plot(mo, plot_config, show_plots):
	equilibrium_figure = show_plots(0.6, 0.5, regime="Equilibrium")
	mo.ui.plotly(equilibrium_figure, config=plot_config)
	return


@app.cell
def elimination_heading(mo, sample_availability):
	mo.md(
		"## Elimination candidates\n\n"
		"$c_{\\mathrm{half}} = 0.2$, $\\chi_{cm} = 3.0$. Representative example: **ID 933**."
		"\n\n" + sample_availability(0.2, 3.0)
	)
	return


@app.cell
def elimination_plot(mo, plot_config, show_plots):
	elimination_figure = show_plots(0.2, 3.0, regime="Elimination")
	mo.ui.plotly(elimination_figure, config=plot_config)
	return


@app.cell
def escape_heading(mo, sample_availability):
	mo.md(
		"## Escape candidates\n\n"
		"$c_{\\mathrm{half}} = 0.7$, $\\chi_{cm} = 3.0$. Representative example: **ID 1036**."
		"\n\n" + sample_availability(0.7, 3.0)
	)
	return


@app.cell
def escape_plot(mo, plot_config, show_plots):
	escape_figure = show_plots(0.7, 3.0, regime="Escape")
	mo.ui.plotly(escape_figure, config=plot_config)
	return


@app.cell
def comparison_heading(mo):
	mo.md("""
    ## Representative regimes
    "
       	"
    "
       	"Equilibrium (**112**), elimination (**933**), and escape (**1036**), using the
    "
       	"comparison palette and marker shapes from the original notebook. Spatial axes
    "
       	"retain equal scaling, with tick labels hidden for the side-by-side comparison.
    """)
	return


@app.cell
def comparison_plot(make_sample_figure, mo, plot_config, regime_styles):
	regime_figure = make_sample_figure(
		[
			(112, "Equilibrium<br>ID 112"),
			(933, "Elimination<br>ID 933"),
			(1036, "Escape<br>ID 1036"),
		],
		title="Representative ABM regimes · time 500",
		styles=regime_styles,
		hide_ticks=True,
	)
	# Leave more space above the panel headings and tighten the legend gap.
	regime_figure.update_layout(
		margin={"t": 85, "b": 65},
		legend={"y": 0, "yanchor": "top", "font": {"size": 18}},
	)
	# Anchor each panel heading to its own axes, even after responsive resizing.
	for _index, _annotation in enumerate(regime_figure.layout.annotations, start=1):
		_annotation.update(
			yref="y domain" if _index == 1 else f"y{_index} domain",
			y=1,
			yshift=2,
			borderpad=0,
			font={"size": 18},
		)
	mo.ui.plotly(regime_figure, config=plot_config)
	return


if __name__ == "__main__":
	app.run()

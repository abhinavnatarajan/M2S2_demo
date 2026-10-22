#!/usr/bin/env python
"""Script to compute persistent statistics for synthetic tumor microenvironment dataset."""

from pathlib import Path
from typing import Any

import polars as pl

from utils import generate_stats, make_cmdline_parser, make_filename_index_map

base_path = Path(__file__).parent


if __name__ == "__main__":
	args_override = {
		"Required arguments": {
			"--dataset-dir": {
				"default": str(base_path.joinpath("data", "simulation_samples")),
				"required": False,
			},
			"--output-dir": {
				"default": str(base_path.joinpath("stats")),
				"required": False,
			},
			"--params-file": {
				"action": "store",
				"default": str(base_path.joinpath("data", "simulation_params.csv")),
				"type": str,
				"help": "Path to the CSV file containing the ABM parameters "
				"used to generate each sample.",
			},
			"--eee-labels-file": {
				"action": "store",
				"default": str(base_path.joinpath("data", "ground_truth_labels.csv")),
				"type": str,
				"help": "Path to the CSV file containing the "
				"ground truth classification of the samples.",
			},
		},
		"Dataset-specific options": {
			"--x-column": {"default": "points_x"},
			"--y-column": {"default": "points_y"},
			"--label-column": {"default": "celltypes"},
			"--phenotype-column": {
				"action": "store",
				"default": "phenotypes",
				"type": str,
				"help": "Column containing the phenotype values for macrophages.",
			},
			"--labels-include": {
				"choices": [
					"Tumour",
					"Macrophage",
					"Stroma",
					"Vessel",
					"Necrotic",
				],
			},
			"--labels-exclude": {
				"choices": [
					"Tumour",
					"Macrophage",
					"Stroma",
					"Vessel",
					"Necrotic",
				],
			},
			"--merge-macrophage": {
				"action": "store_true",
				"help": "If set, do not distinguish between M1 and M2 macrophages.",
			},
		},
		"Logging options": {
			"--logfile-dir": {"default": str(base_path.joinpath("logs"))},
		},
	}

	parser = make_cmdline_parser(args_override)
	cmdline_args = parser.parse_args()

	filtration_algorithm = cmdline_args.filtration_algorithm
	params_df = (
		pl.read_csv(cmdline_args.params_file)
		.select("ID", "chi_macrophageToCSF", "halfMaximalExtravasationCsf1Conc")
		.rename(
			{
				"chi_macrophageToCSF": "chi_c^m",
				"halfMaximalExtravasationCsf1Conc": "c_half",
			},
		)
		.cast({"ID": pl.Int64, "chi_c^m": pl.Float64, "c_half": pl.Float64})
	)
	# Merge the parameter data with the labels dataframe using the indexing columns
	labels_df = pl.read_csv(cmdline_args.eee_labels_file).cast(
		{"chi_c^m": pl.Float64, "c_half": pl.Float64},
	)
	meta_df = params_df.join(labels_df, on=["chi_c^m", "c_half"], how="left", maintain_order="left")
	meta_by_id = {row["ID"]: row for row in meta_df.iter_rows(named=True)}

	def _mapping_func(filename: str) -> dict[str, int | str]:
		sample_id = int(filename.split("_", maxsplit=1)[0].removeprefix("ID-"))
		return {
			"filtration_algorithm": filtration_algorithm,
			"label": meta_by_id[sample_id]["label"],
			"ID": sample_id,
		}

	filename_index_map = make_filename_index_map(_mapping_func)

	label_col = cmdline_args.label_column
	phenotype_col = cmdline_args.phenotype_column
	merge_macrophage = cmdline_args.merge_macrophage

	def _data_loader(
		filepath: Path,
		index_dict: dict[str, Any],
	) -> tuple[pl.DataFrame, dict[str, float]]:
		data_df = pl.read_csv(filepath, infer_schema_length=None)
		macrophages = data_df[label_col] == "Macrophage"
		m1_macrophages = macrophages & (data_df[phenotype_col] < 0.5)  # noqa: PLR2004
		m2_macrophages = macrophages & (data_df[phenotype_col] >= 0.5)  # noqa: PLR2004
		m2_ratio = m2_macrophages.sum() / macrophages.sum() if macrophages.sum() else float("nan")
		if not merge_macrophage:
			data_df = data_df.with_columns(
				pl.when(m1_macrophages)
				.then(pl.lit("M1_Macrophage"))
				.when(m2_macrophages)
				.then(pl.lit("M2_Macrophage"))
				.otherwise(pl.col(label_col))
				.alias(label_col),
			)
		sample_id = index_dict["ID"]
		row = meta_by_id[sample_id]
		c_half = row["c_half"]
		chi = row["chi_c^m"]
		metadata_dict: dict[str, float] = {"M2_ratio": m2_ratio, "c_half": c_half, "chi_c^m": chi}
		return (data_df, metadata_dict)

	generate_stats(cmdline_args, filename_index_map, _data_loader)

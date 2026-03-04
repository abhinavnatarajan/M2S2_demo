#!/usr/bin/env python
"""Script to compute persistent statistics for Dowker multiplex dataset."""

from pathlib import Path
from typing import Any

from pandas import DataFrame, read_csv

from utils import generate_stats, make_cmdline_parser, make_filename_index_map


def main() -> None:  # noqa: D103
    args_override = {
        "Required arguments": {
            "--dataset-dir": {
                "default": "dowker_comparison/data/samples",
                "required": False,
            },
            "--output-dir": {
                "default": "dowker_comparison/stats",
                "required": False,
            },
            "--params-file": {
                "action": "store",
                "nargs": "?",
                "default": "dowker_comparison/data/data_params.csv",
                "type": str,
                "help": "Path to the CSV file containing the ABM parameters "
                "used to generate each sample.",
            },
            "--eec-labels-file": {
                "action": "store",
                "nargs": "?",
                "default": "dowker_comparison/data/eec_labels.csv",
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
                "nargs": "?",
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
            "--logfile-dir": {"default": "dowker_comparison/logs"},
        },
    }

    parser = make_cmdline_parser(args_override)
    cmdline_args = parser.parse_args()

    filtration_algorithm = cmdline_args.filtration_algorithm
    params_df = (
        read_csv(cmdline_args.params_file)
        .loc[:, ["ID", "chi_macrophageToCSF", "halfMaximalExtravasationCsf1Conc"]]
        .rename(
            columns={
                "chi_macrophageToCSF": "chi_c^m",
                "halfMaximalExtravasationCsf1Conc": "c_half",
            },
        )
        .astype({"ID": "int", "chi_c^m": "float", "c_half": "float"})
    )
    # Merge the parameter data with the labels dataframe using the indexing columns
    labels_df = (
        read_csv(cmdline_args.eec_labels_file)
        .astype(
            {"chi_c^m": "float", "c_half": "float"},
        )
    )
    meta_df = params_df.merge(labels_df, on=["chi_c^m", "c_half"], how="left")
    meta_df.set_index("ID")

    def _mapping_func(filename: str) -> dict[str, int | str]:
        sample_id = int(filename.split("_")[0].removeprefix("ID-"))
        return {
            "filtration_algorithm": filtration_algorithm,
            "label": meta_df.loc[sample_id, "label"],
            "ID": sample_id,
        }

    filename_index_map = make_filename_index_map(_mapping_func)

    label_col = cmdline_args.label_column
    phenotype_col = cmdline_args.phenotype_column
    merge_macrophage = cmdline_args.merge_macrophage

    def data_loader(
        filepath: Path,
        index_dict: dict[str, Any],
    ) -> tuple[DataFrame, dict[str, float]]:
        data_df = read_csv(filepath, index_col=0)
        macrophages = data_df[label_col] == "Macrophage"
        m1_macrophages = macrophages & (data_df[phenotype_col] < 0.5)
        m2_macrophages = macrophages & (data_df[phenotype_col] >= 0.5)
        m2_ratio = (m2_macrophages.sum() / macrophages.sum()).item()
        if not merge_macrophage:
            data_df.loc[m1_macrophages, label_col] = "M1_Macrophage"
            data_df.loc[m2_macrophages, label_col] = "M2_Macrophage"
        sample_id = index_dict["ID"]
        row = meta_df.loc[sample_id]
        c_half = row["c_half"].item()
        chi = row["chi_c^m"].item()
        metadata_dict: dict[str, float] = {"M2_ratio": m2_ratio, "c_half": c_half, "chi_c^m": chi}
        return (data_df, metadata_dict)

    generate_stats(cmdline_args, filename_index_map, data_loader)


if __name__ == "__main__":
    main()

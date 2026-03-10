#! /usr/bin/env python
"""Script to compute persistent statistics for the adenoma-carcinoma dataset."""

from pathlib import Path
from typing import Any

from utils import generate_stats, make_cmdline_parser, make_filename_index_map

base_path = Path(__file__).parent


if __name__ == "__main__":
    args_override = {
        "Required arguments": {
            "--dataset-dir": {
                "default": str(base_path.joinpath("data", "ROIs")),
                "required": False,
            },
            "--output-dir": {
                "default": str(base_path.joinpath("stats")),
                "required": False,
            },
        },
        "Dataset-specific options": {
            "--label-column": {"default": "Celltype"},
            "--labels-include": {
                "choices": [
                    "Neutrophil",
                    "Macrophage",
                    "Cytotoxic T Cell",
                    "T Helper Cell",
                    "Treg Cell",
                    "Epithelium (imm)",
                    "Periostin",
                    "CD146",
                    "CD34",
                    "SMA",
                    "Podoplanin",
                    "Epithelium (str)",
                ],
            },
            "--labels-exclude": {
                "choices": [
                    "Neutrophil",
                    "Macrophage",
                    "Cytotoxic T Cell",
                    "T Helper Cell",
                    "Treg Cell",
                    "Epithelium (imm)",
                    "Periostin",
                    "CD146",
                    "CD34",
                    "SMA",
                    "Podoplanin",
                    "Epithelium (str)",
                ],
            },
        },
        "Logging options": {
            "--logfile-dir": {"default": str(base_path.joinpath("logs"))},
        },
    }

    parser = make_cmdline_parser(args_override)
    cmdline_args = parser.parse_args()

    def _mapping_func(filename: str) -> dict[str, Any]:
        tokens = filename.split("_")
        patient_id = int(tokens[0])
        sample_id = int(tokens[-1].replace("ID-", ""))
        sample_type = tokens[1]
        return {
            "patient_id": patient_id,
            "sample_type": sample_type,
            "sample_id": sample_id,
            "filtration_algorithm": cmdline_args.filtration_algorithm,
        }

    filename_index_map = make_filename_index_map(_mapping_func)
    generate_stats(cmdline_args, filename_index_map)

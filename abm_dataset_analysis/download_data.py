#!/usr/bin/env python3
"""Download the ABM-generated synthetic tumor microenvironment dataset for analysis."""

import shutil
from pathlib import Path

from utils.download_utils import download_file, extract_zip


def main() -> None:
    """Download the zipped CSV files containing the point cloud data."""
    print()
    curdir = Path(__file__).parent
    data_path = curdir.joinpath("data")

    zip_url = "https://github.com/JABull1066/MacrophageSensitivityABM/releases/download/2-param-data/all2Params_t500.zip"
    download_file(zip_url, data_path.joinpath("data.zip"), verbose=True, show_progress=True)
    extract_zip(data_path.joinpath("data.zip"), data_path)
    data_path.joinpath("data.zip").unlink()
    if data_path.joinpath("simulation_samples").exists():
        # delete existing directory if it exists
        shutil.rmtree(data_path.joinpath("simulation_samples"))
    Path(data_path.joinpath("17082022_all2Params_t500")).rename(
        data_path.joinpath("simulation_samples"),
    )

    params_file_url = "https://github.com/JABull1066/MacrophageSensitivityABM/releases/download/2-param-data/params_2ParamSweep.csv"
    download_file(params_file_url, data_path.joinpath("simulation_params.csv"))
    print("Done.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Download the colorectal cancer dataset for analysis."""

import shutil
from pathlib import Path

from utils.download_utils import download_file, extract_zip


def main() -> None:
    """Download the zipped CSV files containing the point cloud data."""
    print()
    curdir = Path(__file__).parent
    data_path = curdir.joinpath("data")
    zip_url = "https://github.com/JABull1066/SHIFT-Score-Carcinoma-in-Adenoma/releases/download/v1.0/AllPointcloudsAsCSVs_central.zip"
    download_file(zip_url, data_path.joinpath("data.zip"), verbose=True, show_progress=True)
    extract_zip(data_path.joinpath("data.zip"), data_path)
    data_path.joinpath("data.zip").unlink()
    if data_path.joinpath("ROIs").exists():
        shutil.rmtree(data_path.joinpath("ROIs"))
    Path(data_path.joinpath("AllPointcloudsAsCSVs_central")).rename(data_path.joinpath("ROIs"))
    if data_path.joinpath("coordinates.csv").exists():
        data_path.joinpath("coordinates.csv").unlink()
    Path(data_path.joinpath("AllPointcloudsAsCSVs_central_coordinates.csv")).rename(
        data_path.joinpath("coordinates.csv"),
    )
    print("Done.")


if __name__ == "__main__":
    main()

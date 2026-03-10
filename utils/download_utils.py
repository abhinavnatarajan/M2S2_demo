"""Utility functions for downloading and extracting files with progress bars."""

import logging
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(levelname)s: %(message)s", level="INFO")


def download_file(
    url: str,
    output_path: Path,
    *,
    verbose: bool = True,
    show_progress: bool = True,
) -> None:
    """Download a file from a given URL with a progress bar."""
    Path.mkdir(output_path.parent, parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response:  # noqa: S310
        total = int(response.headers.get("Content-Length", 0))
        if verbose:
            logger.info("Download URL: %s", url)
            logger.info("Download path: %s", output_path)
            if total:
                size = float(total)
                for unit in ("B", "KB", "MB", "GB", "TB"):
                    if size < 1e3:
                        logger.info("File size: %.2f %s", size, unit)
                        break
                    size /= 1e3

        with (
            Path.open(output_path, "wb") as f,
            tqdm(
                total=total or None,
                unit="B",
                unit_scale=True,
                desc="Downloading",
                disable=not show_progress,
            ) as pbar,
        ):
            while chunk := response.read(8192):
                f.write(chunk)
                pbar.update(len(chunk))


def extract_zip(
    zip_path: Path,
    extract_to: Path,
    *,
    verbose: bool = True,
    show_progress: bool = True,
) -> None:
    """Extract a zip file with a progress bar."""
    if verbose:
        logger.info("Extracting %s into %s", zip_path, extract_to)
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.infolist()
        with tqdm(
            total=len(members),
            unit="file",
            desc="Extracting",
            disable=not show_progress,
        ) as pbar:
            for member in members:
                zf.extract(member, extract_to)
                pbar.update(1)

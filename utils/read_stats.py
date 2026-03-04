"""Functions for reading computed statistics from disk during classification."""

from itertools import product
from os import PathLike
from pathlib import Path
from typing import Literal

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

__all__ = ["read_stats"]


def read_stats(
    stat_dir: PathLike,
    table_format: Literal["parquet", "pyarrow", "pandas"] = "parquet",
    **kwargs,  # noqa: ANN003
) -> pq.ParquetDataset | pa.Table | pd.DataFrame:
    """Read the statistics computed by `generate_stats_pairs` from a specified directory.

    This function allows for reading the computed statistics in different formats,
    supporting "parquet" for raw ParquetDataset objects, "pyarrow" for PyArrow Table objects,
    and "pandas" for pandas DataFrame objects. This flexibility is useful for different downstream
    analysis tasks which may require specific data formats.

    Parameters
    ----------
    stat_dir : Pathlike
            The directory from which to read the computed statistics.
            This should be the same directory specified
            as the `output_dir` in `generate_stats_pairs`.
    table_format : (Literal["parquet", "pyarrow", "pandas"], optional)
            The format in which to read the statistics. Defaults to "parquet".
    **kwargs:
            Additional keyword arguments to pass to the `pyarrow.dataset.dataset` function.

    Returns
    -------
    pa.ParquetDataset | pa.Table | DataFrame:
            The computed statistics in the specified format.
            The exact type of the return value depends on the `format` parameter.

    Raises
    ------
    ValueError:
            If an invalid format is specified.

    Examples
    --------
    ```python
    stats_df = read_stats_pairs("/path/to/stats", format="pandas")
    print(stats_df.head())
    ```

    """
    stat_path = Path(stat_dir).resolve()
    dataset = pq.ParquetDataset(stat_path, **kwargs)
    if table_format == "parquet":
        return dataset
    dataset = dataset.read()
    if table_format == "pyarrow":
        return dataset
    dataset = dataset.to_pandas()
    if table_format == "pandas":
        return dataset
    errmsg = "Invalid format specified. Use 'parquet', 'pyarrow', or 'pandas'."  # pyright: ignore[reportUnreachable]
    raise ValueError(errmsg)


def stat_name_split(name: str) -> tuple[str, int, str]:
    """Split a stat name of the form '<dgm><dim>-<statistic>' into a tuple."""
    dgm_name_and_dim, statistic = name.split("-")
    dgm_name, dgm_dim = dgm_name_and_dim[:-1], int(dgm_name_and_dim[-1])
    return (dgm_name, dgm_dim, statistic)


def normalize_colname(name: tuple[str, str]) -> tuple[str, int, str, tuple[str, ...]]:
    """Canonicalise column headers.

    Convert a column index of the form
    (<stat name>, <cell_group>)
    to
    (<dgm>,<dim>,<statistic>,<cell_group>).
    """
    stat_name = name[0]
    cod_types = name[1]
    return (*stat_name_split(stat_name), tuple(cod_types.split("/")))


def discard_feature(dgm: str, dim: int, statistic: str, cell_group: tuple[str, ...]) -> bool:
    """Filter out unwanted features."""
    imm_panel_types = (
        "Epithelium (imm)",
        "Neutrophil",
        "Macrophage",
        "Cytotoxic T Cell",
        "Treg Cell",
        "T Helper Cell",
    )
    str_panel_types = ("Epithelium (str)", "CD146", "CD34", "Periostin", "SMA", "Podoplanin")
    cross_pairs = tuple(
        tuple(sorted(pair)) for pair in product(("Epithelium (imm)",), str_panel_types)
    ) + tuple(tuple(sorted(pair)) for pair in product(("Epithelium (str)",), imm_panel_types))
    return (
        dgm in ("cod", "rel")  # discard codomain and relative diagrams
        or (
            dim == 0  # dimension 0
            and dgm == "cok"  # cokernel in dimension 0 is always empty
        )
        or (
            dim == 0  # dimension 0
            and dgm in ("dom", "im")
            and (any(x in statistic for x in ("birth", "midpt", "length")))
            # births are always 0 in dom0 and im0
            # midpts and length not required since we have death
        )
        or (
            "iqr" in statistic  # drop interquartile range since we have p25 and p75
        )
        or (
            # ignore pairs where the cell types are not in the same panel
            len(cell_group) == 2  # noqa: PLR2004
            and tuple(sorted(cell_group)) in cross_pairs
        )
        or (
            len(cell_group) == 1  # single cell type
            and dgm != "dom"  # keep only the domain
        )
        or (
            len(cell_group) > 1  # more than one cell type
            and dgm == "dom"  # don't need the domain diagrams for pairs or triples
        )
    )

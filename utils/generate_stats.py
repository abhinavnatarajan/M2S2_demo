"""Functions to generate persistent statistics from the datasets."""

import argparse
import logging
import os
import pprint
import sys
import time
from collections.abc import Callable, Collection, Iterator, Mapping
from copy import deepcopy
from functools import partial
from itertools import chain, combinations
from os import cpu_count
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from chalc import chromatic
from chalc.sixpack import KChromaticInclusion, KChromaticQuotient
from pandas import DataFrame, Series, read_csv
from tqdm import tqdm

from .logging import configure_logger
from .parallel import PoolWithLogger
from .persistent_stats import get_pers_stats, pers_stats_names

__all__ = ["generate_stats", "make_cmdline_parser", "make_filename_index_map"]


def default_data_loader(
    filepath: Path,
    _: dict[str, Any],
) -> tuple[DataFrame, dict[str, Any]]:
    return (read_csv(filepath, header=0, index_col=False), {})


def generate_stats(
    cmdline_args: argparse.Namespace,
    filename_index_map: Mapping[str, dict[str, Any]],
    data_loader: Callable[
        [Path, dict[str, Any]],
        tuple[DataFrame, dict[str, Any]],
    ] = default_data_loader,
) -> int:
    """Compute persistent statistics for all pairs of labels from datasets of labelled 2D data.

    This function computes the persistent statistics
    from the 6-pack of persistent homology diagrams
    computed using chromatic Delaunay filtrations,
    from a collection of datasets of labelled 2D data.
    The datasets need to be in a directory, and each dataset
    should be a CSV file representing a labelled point cloud.
    The files must have columns for the x and y coordinates
    and labels of the labelled point cloud.
    A set of labels L to consider is passed as an input parameter.
    For each sample, the persistent statistics are computed for the 6-pack
    corresponding to the k-chromatic quotient map.
    These statistics are saved to an Apache Parquet dataset with partitioning (see https://parquet.apache.org/).

    Parameters
    ----------
    cmdline_args: argparse.Namespace
        A namespace or an object with the following attributes:
        - dataset_dir (str): The directory containing the datasets.
            Each dataset should be a CSV file.
        - files_list (list | None): A list of file paths to process,
            which must be relative to the dataset_dir.
            If None, all files in the dataset_dir will be processed.
        - output_dir (str): The directory where the computed statistics will be
            saved as an Apache Parquet dataset.
        - resume (bool): If True, existing statistics will be not be recomputed.
            If False, existing statistics will be recomputed and overwritten.
        - x_column (str): The name of the column in the CSV files
            that contains the x coordinates of the points.
        - y_column (str): The name of the column in the CSV files
            that contains the y coordinates of the points.
        - label_column (str): The name of the column in the CSV files
            that contains the labels of the points.
        - labels_include (list[str]) : The labels to include in the analysis.
        - labels_exclude (list[str]) : The labels to exclude in the analysis.
        - max_num_labels (int): The maximum number of labels to consider per combination.
        - min_num_labels (int): The minimum number of labels to consider per combination.
        - max_diagram_dimension (int): The maximum dimension of
            the persistence diagrams to compute.
        - filtration_algorithm (str): The method used for computing
            the chromatic Delaunay filtration.
        - min_count (int): The minimum number of points required
            for a label to be included in the analysis.
        - num_workers (int): The number of worker processes to use
            for parallel computation.
        - verbosity (int): The logging verbosity level.
        - logfile_dir (str): The directory where the log file will be saved.
        - disable_progressbar (bool): If True, the progress bar will be disabled.
        - test_run (bool): If True, the function will run in test mode,
            processing only a small number of files with a reduced number of workers.
        - dry_run (bool): If True, the function will not compute any statistics.
            Only the list of files to be processed will be output to the logs.
    filename_index_map : Mapping[str, dict[str,Any]]
        A mapping that takes a filename (without path or filetype suffix)
        to a dictionary of metadata describing the dataset in the file.
        The metadata must be unique between files, as it used for indexing.
        This dictionary will be added to the corresponding row in the output dataset.
    data_loader : Callable[[Path, dict[str, Any]], tuple[DataFrame, dict[str,Any]]], optional
        A function that takes a file path and the index dictionary,
        and returns a DataFrame and a dictionary of metadata.
        The metadata dictionary can be used to add additional information to the stored statistics.
        Used to load the data from the CSV files (and to pre-process the dataframe if needed).
        By default, it reads the CSV file into a DataFrame and returns an empty metadata dictionary.

    Returns
    -------
    None:
        The function saves the computed statistics to an Apache Parquet dataset
        in the directory specified by `output_dir`.

    """
    num_workers: int = cmdline_args.num_workers
    if num_workers <= 0:
        num_workers = cpu_count() or 4
    test_run: bool = cmdline_args.test_run
    if test_run:
        num_workers = min(4, num_workers)
    logfile_dir: str = cmdline_args.logfile_dir
    verbosity: str = cmdline_args.verbosity
    resume: bool = cmdline_args.resume
    dry_run: bool = cmdline_args.dry_run
    disable_progressbar: bool = cmdline_args.disable_progressbar

    logger = init_logging(
        logging.getLogger(__name__),
        "generate_stats",
        logfile_dir,
        verbosity,
    )
    logger.debug("Provided arguments:\n%s", pprint.pformat(vars(cmdline_args)))

    # Validate arguments
    validate_args(cmdline_args, logger)

    files_to_process, filename_index_map = get_files_to_process(
        cmdline_args,
        filename_index_map,
        logger,
    )
    index_df = DataFrame(list(filename_index_map.values()))
    max_partitions = max(index_df[col].nunique() for col in index_df.columns)
    if dry_run:
        logger.info("Dry run complete.")
        sys.exit(0)

    existing_data_behavior = "overwrite_or_ignore" if resume else "delete_matching"
    output_path = Path(cmdline_args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Number of files to process: %d", len(files_to_process))
    logger.info("Number of workers: %d", num_workers)
    logger.info("Begin processing...")

    with PoolWithLogger(
        processes=num_workers,
        logger=logger,
        initargs=(),
    ) as parallel:
        for filepath in tqdm(
            files_to_process,
            desc="Files",
            bar_format="{l_bar}{bar:60}{r_bar}",
            disable=disable_progressbar,
            total=len(files_to_process),
        ):
            results = process_file(
                filepath,
                filename_index_map,
                data_loader,
                cmdline_args,
                parallel,
                logger,
            )
            results_df = DataFrame(results)
            index_in_df = filename_index_map[filepath.stem]
            for col in index_in_df:
                results_df[col] = results_df[col].astype("category")
            results_table = pa.Table.from_pandas(results_df, preserve_index=False)
            pq.write_to_dataset(
                results_table,
                root_path=output_path,
                partition_cols=index_in_df.keys(),
                max_partitions=max_partitions,
                existing_data_behavior=existing_data_behavior,
                compression="gzip",
            )
            logger.debug("Saved stats for %s", filepath.name)

    logger.info("Finished")
    sys.exit(0)


def process_file(
    filepath: Path,
    filename_index_map: Mapping[str, dict],
    data_loader: Callable[[Path, dict[str, Any]], tuple[DataFrame, dict]],
    cmdline_args: argparse.Namespace,
    par_pool: PoolWithLogger,
    logger: logging.Logger,
) -> list[dict]:
    logger.debug("Reading %s...", filepath)

    x_col: str = cmdline_args.x_column
    y_col: str = cmdline_args.y_column
    label_col: str = cmdline_args.label_column
    labels_include: list[str] = cmdline_args.labels_include
    labels_exclude: list[str] = cmdline_args.labels_exclude
    min_count: int = cmdline_args.min_count
    filtration_algorithm: Literal["alpha", "delaunay_cech", "delaunay_rips"] = (
        cmdline_args.filtration_algorithm
    )
    max_diagram_dimension: int = cmdline_args.max_diagram_dimension
    max_num_labels: int = cmdline_args.max_num_labels
    min_num_labels: int = cmdline_args.min_num_labels

    index_dict = filename_index_map[filepath.stem]
    data_df, metadata = data_loader(filepath, index_dict)
    data_df = data_df.loc[:, [x_col, y_col, label_col]]
    data_df[label_col] = data_df[label_col].astype("category")
    if len(labels_include) > 0:
        data_df = data_df.loc[data_df[label_col].isin(labels_include)]
    if len(labels_exclude) > 0:
        data_df = data_df.loc[~data_df[label_col].isin(labels_exclude)]
    labels_retain: list[str] = sorted(
        (
            str(x)
            for x in data_df[label_col].value_counts().loc[lambda c: c >= min_count].index.tolist()
        ),
    )
    logger.debug("Begin computing stats for %s.", filepath.name)
    metadata = metadata | index_dict
    if len(labels_retain) <= min_num_labels:
        logger.debug(
            "Not enough labels retained in %s, will add empty row...",
            filepath.name,
        )
        return [
            metadata | {"codomain": None},
        ]
    data_df = data_df.loc[data_df[label_col].isin(labels_retain)]
    logger.debug(
        "Retaining the following labels for %s:\n%s",
        filepath.name,
        labels_retain,
    )
    label_combinations = list(
        chain.from_iterable(
            combinations(labels_retain, num_labels_codomain)
            for num_labels_codomain in range(
                min_num_labels,
                max_num_labels + 1,
            )
        ),
    )
    func = partial(
        get_stats_k_chromatic_quotient,
        data_df,
        x_col,
        y_col,
        label_col,
        filtration_algorithm,
        max_diagram_dimension,
    )
    gather = list(
        tqdm(
            par_pool.imap_unordered(
                func=func,
                iterable=label_combinations,
            ),
            total=len(label_combinations),
            desc="Label combinations",
            bar_format="{l_bar}{bar:50}{r_bar}",
            leave=False,
            disable=cmdline_args.disable_progressbar,
        ),
    )
    results = [metadata | one_combination_stats for one_combination_stats in gather]
    logger.debug("Finished computing stats for %s", filepath.name)
    return results


def make_cmdline_parser(overrides: dict[str, dict]) -> argparse.ArgumentParser:
    """Get the specification of arguments required for generating k-chromatic quotient stats."""
    args = {
        "Required arguments": {
            "--dataset-dir": {
                "action": "store",
                "type": str,
                "required": True,
                "help": "The datasets need to be in a directory, "
                "and each dataset should be a CSV file representing a labelled point cloud.\n"
                "The files must have columns for the x and y coordinates and "
                "labels of the labelled point cloud.",
            },
            "--output-dir": {
                "action": "store",
                "type": str,
                "required": True,
                "help": "The root directory where the statistics will be saved. "
                "The statistics will be saved in Apache Parquet format "
                "with hive-style partitioning.",
            },
        },
        "Processing options": {
            "--files-list": {
                "action": "store",
                "nargs": "*",
                "default": [],
                "type": str,
                "help": "A list of filenames to process. "
                "If not provided, all files in the dataset directory will be processed.",
            },
            "--resume": {
                "action": argparse.BooleanOptionalAction,
                "default": True,
                "help": "If set, existing statistics will not be recomputed. "
                "Otherwise, existing statistics will be recomputed and overwritten. "
                "Useful for large datasets and long runs, since results can be persisted "
                "between runs in case the run is stopped for some reason.",
            },
            "--num-workers": {
                "action": "store",
                "nargs": "?",
                "default": 0,
                "type": int,
                "help": "Number of CPUs to use for parallel processing. "
                "Set to 0 to use all available CPUs.",
            },
        },
        "Dataset-specific options": {
            "--x-column": {
                "action": "store",
                "default": "x",
                "type": str,
                "help": "The column name for the x-coordinate.",
            },
            "--y-column": {
                "action": "store",
                "default": "y",
                "type": str,
                "help": "The column name for the y-coordinate.",
            },
            "--label-column": {
                "action": "store",
                "default": "label",
                "type": str,
                "help": "The column giving the label of the point.",
            },
            "--labels-include": {
                "action": "store",
                "nargs": "*",
                "default": [],
                "type": str,
                "help": "The labels to include in the computations. "
                "If provided, labels not in this list will be ignored.",
            },
            "--labels-exclude": {
                "action": "store",
                "nargs": "*",
                "default": [],
                "type": str,
                "help": "The labels to exclude in the computations. "
                "If provided, labels in this list will be ignored.",
            },
        },
        "Algorithm options": {
            "--max-num-labels": {
                "action": "store",
                "default": 3,
                "type": int,
                "help": "The maximum number of labels to consider per combination.",
            },
            "--min-num-labels": {
                "action": "store",
                "default": 1,
                "type": int,
                "help": "The minimum number of labels to consider per combination.",
            },
            "--max-diagram-dimension": {
                "action": "store",
                "default": 1,
                "type": int,
                "choices": (0, 1),
                "help": "The maximum dimension of the persistence diagrams to compute.",
            },
            "--filtration-algorithm": {
                "action": "store",
                "default": "delaunay_cech",
                "type": str,
                "choices": ("delaunay_rips", "delaunay_cech", "alpha"),
                "help": "The method to use for computing the persistent homology diagrams.",
            },
            "--min-count": {
                "action": "store",
                "default": 3,
                "type": int,
                "help": "If any label class has fewer than this many representatives "
                "in a given sample, it is discarded in that sample.",
            },
        },
        "Logging options": {
            "--verbosity": {
                "action": "store",
                "default": "INFO",
                "type": str,
                "choices": ("DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"),
                "help": "Set the verbosity level of the logger. "
                "Only messages of this level and higher will be logged.",
            },
            "--logfile-dir": {
                "action": "store",
                "type": str,
                "help": "Directory where log files will be saved.",
            },
        },
        "Miscellaneous options": {
            "--disable-progressbar": {
                "action": "store_true",
                "help": "Disable progress bar output.",
            },
            "--test-run": {
                "action": "store_true",
                "help": "Test with small number of workers and trials (ignores some other options)",
            },
            "--dry-run": {
                "action": "store_true",
                "help": "If set, no statistics will be computed but the files to process "
                "will be computed and written to logs.",
            },
        },
    }
    args = deep_merge_dicts(args, overrides)
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    for group_name, group_args in args.items():
        group = parser.add_argument_group(group_name)
        for arg_name, arg_params in group_args.items():
            group.add_argument(arg_name, **arg_params)
    return parser


def validate_args(args: argparse.Namespace, logger: logging.Logger) -> None:
    if args.min_num_labels < 1:
        errmsg = "Minimum number of labels must be positive."
        logger.error(errmsg)
        sys.exit(1)
    if args.max_num_labels < 1:
        errmsg = "Maximum number of labels must be positive."
        logger.error(errmsg)
        sys.exit(1)
    if args.max_num_labels < args.min_num_labels:
        errmsg = (
            "Maximum number of labels must be greater than or equal to "
            "the minimum number of labels."
        )
        logger.error(errmsg)
        sys.exit(1)
    if args.min_count < 0:
        logger.warning("Argument 'min_count' is negative, will not filter anything.")

    def validate_directory(path: str) -> None:
        """Check if the directory exists."""
        p = Path(path).resolve()
        if not p.exists:
            msg = f"Path does not exist: {path}"
            logger.error(msg)
            sys.exit(1)

    validate_directory(args.dataset_dir)
    validate_directory(args.output_dir)


def deep_merge_dicts(base_dict: dict, override_dict: dict) -> dict:
    """Deep merge two dictionaries, with override_dict values taking precedence.

    Parameters
    ----------
    base_dict : dict
        The base dictionary to merge into.
    override_dict : dict
        The dictionary whose values will override base_dict values.

    Returns
    -------
    dict
        A new dictionary with values from both dictionaries, with override_dict
        values taking precedence when keys conflict.

    """
    result = deepcopy(base_dict)

    for key, value in override_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = deepcopy(value)

    return result


def make_filename_index_map(
    mapping_func: Callable[[str], dict[str, Any]],
) -> Mapping[str, dict[str, Any]]:
    """Make a mapping from filenames to metadata dictionaries."""

    class FilenameToMetadataMapping(Mapping[str, dict]):
        """Memoized filename to metadata transformer.

        Implements Mapping interface and caches results of filename transformations.
        File names should be of the form <patient_id>_<sample_type>_ID-<sample_id>.csv
        """

        def __init__(self) -> None:
            self._mapping_func = mapping_func
            self._cache: dict[str, dict[str, Any]] = {}

        def __getitem__(self, filename: str) -> dict[str, Any]:
            """Convert filename to metadata dictionary with memoization.

            Args:
                filename: File name to transform

            Returns:
                A dictionary of indexing/unique metadata about the file.

            """
            if filename not in self._cache:
                self._cache[filename] = self._mapping_func(filename)
            return self._cache[filename]

        def __iter__(self) -> Iterator[str]:
            return iter(self._cache)

        def __len__(self) -> int:
            return len(self._cache)

    return FilenameToMetadataMapping()


def get_stats_k_chromatic_quotient(
    df: DataFrame,
    x_col: str,
    y_col: str,
    label_col: str,
    filtration_algorithm: Literal["alpha", "delaunay_cech", "delaunay_rips"],
    max_diagram_dimension: int,
    codomain_types: tuple[str, ...],
) -> dict[str, tuple[str, ...] | np.float64]:
    codomain_types = tuple(set(codomain_types))
    num_labels = len(codomain_types)
    k = num_labels - 1 if num_labels > 1 else 1
    # Pull out points that are in the codomain
    sub_df: DataFrame = df.loc[df[label_col].isin(codomain_types)]
    points = sub_df[[x_col, y_col]].to_numpy().transpose()
    # Re-index with integers
    colours = sub_df[label_col].cat.codes.to_numpy()
    # Canonicalize the codomain types
    colours = canonicalize_labels(colours)
    # Pre-compute the filtration
    filt = getattr(chromatic, filtration_algorithm)(points, colours)
    logging.debug("Computing stats for codomain %s", codomain_types)  # noqa: LOG015
    dgms = (KChromaticInclusion(filt, 1) if k == 1 else KChromaticQuotient(filt, k)).sixpack()
    barcodes = {
        name: dgms.get_matrix(name, list(range(max_diagram_dimension + 1))) for name in dgms
    }
    stats: dict[str, tuple[str, ...] | np.float64] = {
        "codomain": codomain_types,
    } | get_stats_from_barcodes_dict(barcodes)
    logging.debug("Computed stats for codomain %s", codomain_types)  # noqa: LOG015
    return stats


def canonicalize_labels(v: list[int]) -> np.ndarray[tuple[int], np.dtype[np.int_]]:
    label_set = set(v)
    old_to_new = {label: new_label for new_label, label in enumerate(label_set)}
    return np.array([old_to_new[label] for label in v], dtype=np.int_)


def get_stats_from_barcodes_dict(barcodes: dict[str, list[np.ndarray]]) -> dict[str, np.float64]:
    stats: dict[str, np.float64] = {}
    for diagram_name, barcodes_all_dims in barcodes.items():
        for dim, barcode in enumerate(barcodes_all_dims):
            stat_names = [f"{diagram_name}{dim}-{stat_name}" for stat_name in pers_stats_names]
            stat_values = get_pers_stats(barcode)
            stats = stats | dict(zip(stat_names, stat_values, strict=True))
    return stats


def init_logging(
    logger: logging.Logger,
    logfile_name: str,
    logfile_dir: str,
    verbosity: str,
) -> logging.Logger:
    logfile_path = Path(logfile_dir).resolve()
    Path.mkdir(logfile_path, parents=True, exist_ok=True)
    logfile = logfile_path.joinpath(
        "{}_{}.log".format(logfile_name, time.strftime("%H:%M:%S-%d%b%Y", time.gmtime())),
    )
    logger = configure_logger(
        logger,
        filename=logfile,
        file_level=logging.DEBUG,
        console_level=verbosity,
    )
    logging.getLogger("phimaker").setLevel(logging.ERROR)
    logging.getLogger("chalc.chromatic").setLevel(logging.ERROR)
    logger.info("Logs will be written to %s", logfile)
    return logger


def resolve_data_filepaths(
    cmdline_args: argparse.Namespace,
    logger: logging.Logger,
) -> list[Path]:
    logger.info("Resolving file paths...")
    files_list: list[str] = cmdline_args.files_list
    dataset_dir: str = cmdline_args.dataset_dir
    test_run: bool = cmdline_args.test_run

    dataset_path = Path(dataset_dir).resolve()
    if files_list:
        resolved_files = sorted(dataset_path.joinpath(file).resolve() for file in files_list)
    else:
        resolved_files = sorted(dataset_path.glob("*.csv"))
        if test_run:
            resolved_files = resolved_files[: min(len(resolved_files), 2)]
    # Check if files exist and are readable CSV files
    resolved_and_checked_files = []
    for file in resolved_files:
        if not file.is_file():
            logger.warning("File %s could not be found!", file.as_uri())
            continue
        if not os.access(file, os.R_OK):
            logger.warning("File %s is not readable!", file.as_uri())
            continue
        if file.suffix != ".csv":
            logger.warning("File %s is not a CSV file!", file.as_uri())
            continue
        resolved_and_checked_files.append(file)
    return resolved_and_checked_files


def check_already_processed(
    files_to_check: list[Path],  # must be non-empty
    filename_index_map: Mapping[str, dict[str, Any]],
    cmdline_args: argparse.Namespace,
    logger: logging.Logger,
) -> list[Path]:
    resume: bool = cmdline_args.resume
    output_dir: str = cmdline_args.output_dir
    output_path = Path(output_dir).resolve()
    files_to_process = set(files_to_check)
    if not files_to_process:
        errmsg = "No files to process after resolving arguments."
        raise FileNotFoundError(errmsg)
    if not resume:
        logger.info("Existing stats will be recomputed and overwritten.")
    else:
        logger.info("Checking for existing stats...")
        dataset_index_cols = list(filename_index_map[files_to_check[0].stem].keys())
        try:
            processed_records: DataFrame = (
                ds.dataset(output_path, format="parquet", partitioning="hive")
                .to_table(columns=dataset_index_cols)
                .to_pandas()
                .drop_duplicates()
            )
        except (FileNotFoundError, pa.ArrowInvalid):
            processed_records = DataFrame([], columns=Series(dataset_index_cols))

        def already_processed(f: Path) -> bool:
            record_to_check = Series(filename_index_map[f.stem])
            return (processed_records == record_to_check).all(axis=1).any()

        if (m := len(processed_records)) > 0:
            logger.info("Found existing stats for %d records.", m)
            logger.info(
                "Existing stats will not be recomputed, see the logs for the skipped files.",
            )
            processed_files = {file for file in files_to_check if already_processed(file)}
            skipped = "\n".join(f.name for f in processed_files)
            logger.debug("Following files will be skipped:\n%s", skipped)
            files_to_process = files_to_process.difference(processed_files)
    return list(files_to_process)


def validate_files_to_process(
    *,
    files_to_process: Collection[Path],
    logger: logging.Logger,
) -> None:
    if len(files_to_process) == 0:
        logger.info("No files to process.")
        sys.exit(0)
    # validate that the files exist
    logger.info("Checking if files to process exist and are readable...")
    for file in files_to_process:
        if not file.is_file():
            logger.error("File %s could not be found!", file.as_uri())
            sys.exit(1)
        if not os.access(file, os.R_OK):
            logger.error("File %s is not readable!", file.as_uri())
            sys.exit(1)
        if file.suffix != ".csv":
            logger.error("File %s is not a CSV file!", file.as_uri())
            sys.exit(1)


def get_files_to_process(
    cmdline_args: argparse.Namespace,
    filename_index_map: Mapping[str, dict[str, Any]],
    logger: logging.Logger,
) -> tuple[list[Path], dict[str, Any]]:
    # Resolve the data directories
    resolved_filepaths = resolve_data_filepaths(cmdline_args, logger)
    # Each file is associated to a record in the dataset
    # We assume that filename_index_map[filename] produces a unique key for the record
    # The key can be multiple columns, not just a single value
    # Each combination of values in dataset_index_cols
    # encodes the stats for a given dataset and using a specified method
    if not resolved_filepaths:
        errmsg = "No CSV files found after resolving arguments."
        logger.error(errmsg)
        sys.exit(1)
    filename_index_map = {file.stem: filename_index_map[file.stem] for file in resolved_filepaths}
    files_to_process = check_already_processed(
        files_to_check=resolved_filepaths,
        filename_index_map=filename_index_map,
        cmdline_args=cmdline_args,
        logger=logger,
    )
    # Check that we have data and it is accessible
    validate_files_to_process(files_to_process=files_to_process, logger=logger)
    return (files_to_process, filename_index_map)

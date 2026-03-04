"""Module to configure a logger with a file and console handler.

This is intended for making it easier to configure loggers for various scripts used in the pipeline.
"""

import os
import re
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Collection
from logging import (
    DEBUG,
    ERROR,
    INFO,
    FileHandler,
    Formatter,
    Logger,
    LogRecord,
    NullHandler,
    StreamHandler,
    captureWarnings,
)
from pathlib import Path

__all__ = ["_build_handler_filter", "configure_logger"]


def configure_logger(
    logger: Logger,
    filename: str | os.PathLike = os.devnull,
    file_level: int | str = INFO,
    console_level: int | str = ERROR,
    console_ignore: str | Collection[str] = [],
) -> Logger:
    """Configure a logger with a file and console handler.

    Args:
            logger: The logger to configure.
            filename: The file to log to. If set to os.devnull,
            the file handler will be a NullHandler.
            file_level: The log level for the file handler.
            console_level: The log level for the console handler.
            console_ignore: A logger name or collection of logger names
            to ignore when logging to the console.

    """
    captureWarnings(capture=True)
    formatter = Formatter("%(asctime)s %(name)s %(levelname)-8s %(message)s")
    formatter.converter = time.gmtime
    console_handler = StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(console_level)
    if isinstance(console_ignore, str):
        console_ignore = (console_ignore,)
    if isinstance(console_ignore, Collection):
        for logger_name in console_ignore:
            console_handler.addFilter(_build_handler_filter(logger_name))
    if filename == os.devnull:
        file_handler = NullHandler()
    else:
        file_handler = FileHandler(filename, mode="w")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(file_level)
    logger.setLevel(DEBUG)
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def _build_handler_filter(logger: str) -> Callable[[LogRecord], bool]:
    def handler_filter(record: LogRecord) -> bool:
        return record.name != logger

    return handler_filter


class LogParser:
    """Parser for log files to extract processing information."""

    def clear(self) -> None:
        self.parsed = {}  # filename -> {'labels': set, 'combinations': dict}
        self.current_file = None
        self.computing_stats = defaultdict(set)  # track what's being computed
        self.computed_stats = defaultdict(set)  # track what's completed
        self._next_line_callback = None

    def __init__(self) -> None:
        self.clear()

    def parse_log_file(self, log_file_path: str) -> None:
        """Parse a log file and return extracted information."""
        with Path(log_file_path).open("r") as f:
            for line in f:
                if self._next_line_callback:
                    self._next_line_callback(line.strip())
                else:
                    self._process_line(line.strip())
        # Last file might not have a "finished" line, so we finalize it
        if self.current_file and self.current_file in self.parsed:
            started = self.parsed[self.current_file]["combinations"]["started"]
            completed = self.parsed[self.current_file]["combinations"]["completed"]
            failed = started - completed
            self.parsed[self.current_file]["combinations"]["failed"] = failed
            self.current_file = None

    def _process_labels_line(self, labels_line: str) -> None:
        labels_match = re.search(r"\[(\S*)\]", labels_line)
        if labels_match:
            labels_str = labels_match.group(1).strip()
            # Parse the labels - they're quoted strings separated by commas
            labels = [label.strip().strip("'\"") for label in labels_str.split(",")]
            if self.current_file:
                self.parsed[self.current_file]["labels"].update(labels)
        self._next_line_callback = None

    def _process_line(self, line: str) -> None:
        """Process a single log line."""
        # Check for file processing start
        file_match = re.search(r"Begin computing stats for (\S+)\.$", line)
        if file_match:
            self.current_file = file_match.group(1)
            if self.current_file not in self.parsed:
                self.parsed[self.current_file] = {
                    "labels": set(),
                    "combinations": {"started": set(), "completed": set(), "failed": set()},
                }
            return

        # Check for retained labels
        if self.current_file and "Retaining the following labels for" in line:
            # Extract labels from the line - they're in a list format
            labels_next_line_match = re.search(
                r"Retaining the following labels for \S+\s*:\s*$",
                line,
            )
            if labels_next_line_match:
                self._next_line_callback = self._process_labels_line
            return

        # Check for computing stats start
        computing_match = re.search(r"Computing stats for codomain \((.+)\)", line)
        if computing_match:
            codomain_str = computing_match.group(1)
            # Parse the codomain tuple - it's quoted strings separated by commas
            codomain_items = [item.strip().strip("'\"") for item in codomain_str.split(",")]
            codomain = tuple(codomain_items)

            if self.current_file:
                self.parsed[self.current_file]["combinations"]["started"].add(codomain)
            return

        # Check for computed stats completion
        computed_match = re.search(r"Computed stats for codomain \((.+)\)", line)
        if computed_match:
            codomain_str = computed_match.group(1)
            # Parse the codomain tuple - it's quoted strings separated by commas
            codomain_items = [item.strip().strip("'\"") for item in codomain_str.split(",")]
            codomain = tuple(codomain_items)

            if self.current_file:
                self.parsed[self.current_file]["combinations"]["completed"].add(codomain)
            return

        # Check for file processing completion
        if self.current_file and f"Finished computing stats for {self.current_file}" in line:
            # Calculate failed combinations for this file
            started = self.parsed[self.current_file]["combinations"]["started"]
            completed = self.parsed[self.current_file]["combinations"]["completed"]
            failed = started - completed
            self.parsed[self.current_file]["combinations"]["failed"] = failed
            self.current_file = None
            return

    def generate_report(self) -> dict:
        """Generate a comprehensive report from parsed data."""
        report = {
            "summary": {
                "total_files": len(self.parsed),
                "files_with_issues": 0,
                "total_combinations_started": 0,
                "total_combinations_completed": 0,
                "total_combinations_failed": 0,
            },
            "files": {},
        }

        for filename, data in self.parsed.items():
            started = len(data["combinations"]["started"])
            completed = len(data["combinations"]["completed"])
            failed = len(data["combinations"]["failed"])

            # Update summary
            report["summary"]["total_combinations_started"] += started
            report["summary"]["total_combinations_completed"] += completed
            report["summary"]["total_combinations_failed"] += failed

            if failed > 0:
                report["summary"]["files_with_issues"] += 1

            # File details
            report["files"][filename] = {
                "labels": sorted(data["labels"]),
                "combinations": {
                    "started": started,
                    "completed": completed,
                    "failed": failed,
                    "failed_combinations": sorted(data["combinations"]["failed"]),
                },
            }

        return report

    def print_report(self, report: dict, only_failed: bool = True) -> None:
        """Print a formatted report."""
        print("=" * 80)
        print("LOG ANALYSIS REPORT")
        print("=" * 80)

        # Summary
        summary = report["summary"]
        print("\nSUMMARY:")
        print(f"  Total files processed: {summary['total_files']}")
        print(f"  Files with failed combinations: {summary['files_with_issues']}")
        print(f"  Total combinations started: {summary['total_combinations_started']}")
        print(f"  Total combinations completed: {summary['total_combinations_completed']}")
        print(f"  Total combinations failed: {summary['total_combinations_failed']}")
        print(
            f"  Success rate: {
                summary['total_combinations_completed']
                / summary['total_combinations_started']
                * 100:.1f}%",
        )

        # File details
        print("\nFILES WITH FAILURES:")
        for filename, data in report["files"].items():
            if data["combinations"]["failed"] == 0 and only_failed:
                continue
            print(f"\n{filename}:")
            print(f"  Labels ({len(data['labels'])}): {', '.join(data['labels'])}")
            print(
                f"  Combinations - Started: {data['combinations']['started']}, "
                f"Completed: {data['combinations']['completed']}, "
                f"Failed: {data['combinations']['failed']}",
            )

            if data["combinations"]["failed"] > 0:
                print("  Failed combinations:")
                for combo in data["combinations"]["failed_combinations"]:
                    print(f"    {combo}")

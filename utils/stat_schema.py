"""Stable diagram names used in the paper's persisted statistics.

These abbreviations are part of our Parquet/feature schema, independent of
chalc's public persistence-module names.
"""

from typing import Literal

from chalc.morphisms import PersistenceModuleName

type DiagramName = Literal["ker", "dom", "im", "cod", "cok", "rel"]

MODULE_TO_DIAGRAM: dict[PersistenceModuleName, DiagramName] = {
	"kernel": "ker",
	"domain": "dom",
	"image": "im",
	"codomain": "cod",
	"cokernel": "cok",
	"relative": "rel",
}

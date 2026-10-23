"""Stable diagram names used in the paper's persisted statistics.

These abbreviations are part of our Parquet/feature schema, independent of
chalc's public persistence-module names.
"""

import re
from typing import ClassVar, Literal, TypeGuard, cast, get_args

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


class FeatureParts:
	"""Lazily parse a feature name into its components."""

	dgm_and_dim: str
	_dgm: DiagramName
	_dim: int
	statistic: str
	codomain: str
	NUM_FEATURE_NAME_PARTS: ClassVar[int] = 3
	DGM_NAMES_REGEX: ClassVar[str] = "|".join(
		re.escape(name) for name in get_args(DiagramName.__value__)
	)

	@classmethod
	def is_diagram_name(cls, name: str) -> TypeGuard[DiagramName]:
		"""Check if a string is a diagram name."""
		return name in get_args(DiagramName.__value__)

	def __init__(self, feature_name: str) -> None:
		"""Parse a feature name.

		Feature names are of the form <diagram><dimension>-<statistic>-<cell_type/cell_type/...>.
		"""
		# Expects feature names of the form <dgm><dim>-<statistic>-<cell_type/cell_type/...>
		# such as ker0-avg_birth-cell1/cell2/cell3
		tokens = feature_name.split("-", maxsplit=2)
		if len(tokens) != self.NUM_FEATURE_NAME_PARTS:
			errmsg = f"Invalid feature name: {feature_name!r}"
			raise ValueError(errmsg)
		self.dgm_and_dim = tokens[0]
		self.statistic = tokens[1]
		self.codomain = tokens[2]

	def _parse_dgm_and_dim(self) -> None:
		if hasattr(self, "_dim") and hasattr(self, "_dgm"):
			return
		match = re.fullmatch(rf"({self.DGM_NAMES_REGEX})(\d+)", self.dgm_and_dim)
		if not match:
			errmsg = f"Invalid diagram and dimension: {self.dgm_and_dim!r}"
			raise ValueError(errmsg)
		self._dgm = cast("DiagramName", match.group(1))
		self._dim = int(match.group(2))

	@property
	def dim(self) -> int:
		"""Dimension of the feature."""
		self._parse_dgm_and_dim()
		return self._dim

	@property
	def dgm(self) -> DiagramName:
		"""Which diagram of the 6-pack the feature was computed from."""
		self._parse_dgm_and_dim()
		return self._dgm

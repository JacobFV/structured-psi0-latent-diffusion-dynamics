from __future__ import annotations

from pydantic import Field

from .base import Strict


class EntityRef(Strict):
    """Opaque runtime identity. NOT a target label, simulator address or product name."""
    id: str = Field(min_length=1, max_length=128)
    version: int = Field(ge=0)

    def key(self) -> tuple[str, int]:
        return (self.id, self.version)

    def __hash__(self):
        return hash(self.key())

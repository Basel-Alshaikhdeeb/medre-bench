"""Abstract base dataset and shared utilities for relation extraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class RelationExample:
    """Unified representation of a single relation extraction example."""

    text: str
    entity1: str
    entity1_type: str
    entity1_start: int
    entity1_end: int
    entity2: str
    entity2_type: str
    entity2_start: int
    entity2_end: int
    label: str
    label_id: int
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseDataset(ABC):
    """Abstract base class for all dataset adapters."""

    @abstractmethod
    def name(self) -> str:
        """Return the dataset registry key."""
        ...

    @abstractmethod
    def num_labels(self) -> int:
        """Return the number of relation classes."""
        ...

    @abstractmethod
    def label_names(self) -> list[str]:
        """Return ordered list of label names (index = label_id)."""
        ...

    @abstractmethod
    def load_split(self, split: str) -> list[RelationExample]:
        """Load and preprocess a dataset split into RelationExamples."""
        ...

    def entity_marker_strategy(self) -> str:
        """Return the default entity marking strategy for this dataset."""
        return "typed_entity_marker_punct"


def apply_entity_markers(
    text: str,
    e1_start: int,
    e1_end: int,
    e1_type: str,
    e2_start: int,
    e2_end: int,
    e2_type: str,
    strategy: str = "typed_entity_marker_punct",
) -> str:
    """Insert entity markers into text around the two entities.

    Handles the case where entity positions may overlap or where e2 comes
    before e1 in the text.
    """
    # Determine which entity comes first in text
    if e1_start <= e2_start:
        first_start, first_end, first_type, first_tag = e1_start, e1_end, e1_type, "E1"
        second_start, second_end, second_type, second_tag = e2_start, e2_end, e2_type, "E2"
    else:
        first_start, first_end, first_type, first_tag = e2_start, e2_end, e2_type, "E2"
        second_start, second_end, second_type, second_tag = e1_start, e1_end, e1_type, "E1"

    if strategy == "typed_entity_marker_punct":
        first_open = f"@ {first_tag}-{first_type.lower()} @ "
        first_close = f" @ /{first_tag}-{first_type.lower()} @ "
        second_open = f"# {second_tag}-{second_type.lower()} # "
        second_close = f" # /{second_tag}-{second_type.lower()} # "
    elif strategy == "typed_entity_marker":
        first_open = f"[{first_tag}-{first_type}] "
        first_close = f" [/{first_tag}-{first_type}] "
        second_open = f"[{second_tag}-{second_type}] "
        second_close = f" [/{second_tag}-{second_type}] "
    elif strategy == "standard":
        first_open = f"[{first_tag}] "
        first_close = f" [/{first_tag}] "
        second_open = f"[{second_tag}] "
        second_close = f" [/{second_tag}] "
    else:
        raise ValueError(f"Unknown entity marker strategy: {strategy}")

    # Build the marked text by inserting markers (process from end to preserve offsets)
    marked = (
        text[:first_start]
        + first_open
        + text[first_start:first_end]
        + first_close
        + text[first_end:second_start]
        + second_open
        + text[second_start:second_end]
        + second_close
        + text[second_end:]
    )
    return marked

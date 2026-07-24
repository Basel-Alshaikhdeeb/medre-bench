"""Load / save BioC JSON and lift its biomedical entities into typed records.

The BioC JSON we consume looks like::

    {
      "documents": [{
        "id": ...,
        "passages": [{
          "offset": 82,
          "text": "...",
          "annotations": [{
            "id": "dis-...",
            "text": "...",
            "offsets": {"start": 82, "end": 116},
            "infons": {"type": "Disease", "category": "dis", "provider": "LISN",
                       "identifier": "MDR:10067918", "identifier_uml": "L3090697"}
          }, ...],
          "relations": []
        }]
      }]
    }

Only annotations whose ``infons.type`` is one of ``Disease``, ``Chemical`` or
``Gene`` (case-insensitive) are lifted; PICO annotations are ignored.
Annotation offsets in the source file are document-level; we rebase them to
passage-relative offsets since sentence splitting operates per passage.

Preserves the raw dictionary structure so the writeback path can append
``relations[]`` entries in place without disturbing anything else.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from medre_bench.datasets.aggregate import _ENTITY_TYPE_MAP


_BIOMEDICAL_TYPES = {"Disease", "Chemical", "Gene"}


@dataclass
class Entity:
    """One deduped biomedical entity within a passage."""

    ann_id: str
    text: str
    canonical_type: str  # CHEMICAL / DISEASE / GENE
    raw_type: str        # Disease / Chemical / Gene
    doc_start: int
    doc_end: int
    passage_start: int
    passage_end: int
    identifier: str | None
    identifier_uml: str | None

    @property
    def dedup_key(self) -> str:
        """Key used to merge same-concept mentions within a sentence."""
        if self.identifier_uml:
            return f"uml:{self.identifier_uml}"
        if self.identifier:
            return f"id:{self.identifier}"
        return f"tt:{self.canonical_type}:{self.text.lower().strip()}"


@dataclass
class Sentence:
    """One sentence inside a passage with its assigned entities."""

    text: str
    passage_start: int  # passage-relative offset of the sentence's first char
    passage_end: int
    entities: list[Entity] = field(default_factory=list)  # deduped


@dataclass
class Passage:
    """A BioC passage plus the sentence + entity records lifted from it."""

    raw: dict  # reference to the original passage dict for output writeback
    text: str
    doc_offset: int  # document-level offset where the passage's text starts
    document_id: str
    entities: list[Entity] = field(default_factory=list)
    sentences: list[Sentence] = field(default_factory=list)


def load_bioc(path: str | Path) -> dict:
    """Return the raw BioC JSON dict; the caller uses this both for parsing
    and for writeback (relations are appended into it in place)."""
    return json.loads(Path(path).read_text())


def save_bioc(raw: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(raw, indent=4))


def _norm_type(raw_type: str) -> str | None:
    """Return the canonical entity type ('CHEMICAL' / 'DISEASE' / 'GENE') for a
    raw annotation type string, or None if it's out of schema. Handles the
    exact strings the BioC file uses ('Disease' / 'Chemical' / 'Gene') as well
    as the case variants and BigBio-style spellings that already appear in
    ``_ENTITY_TYPE_MAP``."""
    if not raw_type:
        return None
    if raw_type in _ENTITY_TYPE_MAP:
        return _ENTITY_TYPE_MAP[raw_type]
    lowered = raw_type.strip().lower()
    for candidate in ("Disease", "Chemical", "Gene"):
        if lowered == candidate.lower():
            return _ENTITY_TYPE_MAP[candidate]
    return None


def extract_entities(raw_passage: dict) -> list[Entity]:
    """Extract biomedical entities from one passage, dropping PICO entries and
    anything else that isn't Disease / Chemical / Gene."""
    passage_offset = int(raw_passage.get("offset", 0))
    out: list[Entity] = []
    for ann in raw_passage.get("annotations", []) or []:
        infons = ann.get("infons") or {}
        raw_type = infons.get("type") or ""
        # Fast path: only biomedical types survive. This also drops PICO
        # annotations (type in {participant, intervention, outcome}).
        if raw_type not in _BIOMEDICAL_TYPES:
            canonical = _norm_type(raw_type)
            if canonical is None:
                continue
        else:
            canonical = _norm_type(raw_type)
            if canonical is None:  # defensive; shouldn't happen
                continue

        offsets = ann.get("offsets") or {}
        try:
            doc_start = int(offsets["start"])
            doc_end = int(offsets["end"])
        except (KeyError, TypeError, ValueError):
            continue

        text = ann.get("text") or ""
        out.append(
            Entity(
                ann_id=str(ann.get("id", "")),
                text=text,
                canonical_type=canonical,
                raw_type=raw_type,
                doc_start=doc_start,
                doc_end=doc_end,
                passage_start=doc_start - passage_offset,
                passage_end=doc_end - passage_offset,
                identifier=infons.get("identifier"),
                identifier_uml=infons.get("identifier_uml"),
            )
        )
    return out


def iter_passages(raw: dict) -> list[Passage]:
    """Materialize a Passage per BioC passage. Only ``entities`` is populated
    here — sentence splitting happens downstream in pair_enumeration."""
    out: list[Passage] = []
    for doc in raw.get("documents", []) or []:
        doc_id = str(doc.get("id", ""))
        for passage in doc.get("passages", []) or []:
            text = passage.get("text") or ""
            offset = int(passage.get("offset", 0))
            out.append(
                Passage(
                    raw=passage,
                    text=text,
                    doc_offset=offset,
                    document_id=doc_id,
                    entities=extract_entities(passage),
                )
            )
    return out

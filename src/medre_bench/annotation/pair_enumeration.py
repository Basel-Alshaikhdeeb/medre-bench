"""Sentence splitting, same-concept dedup, and pair enumeration per passage.

Entrypoint: :func:`enumerate_pairs`. Given a Passage with its extracted
entities, it splits the passage text into sentences, assigns each entity to
the sentence it falls inside, deduplicates same-concept mentions within a
sentence (by UMLS identifier), and emits every ordered ``(e1, e2)`` pair that
matches one of the five aggregate classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from medre_bench.annotation.bioc_io import Entity, Passage, Sentence
from medre_bench.datasets.aggregate import _canonicalize_pair
from medre_bench.datasets.preprocessing import split_into_sentences


@dataclass
class Pair:
    """One (e1, e2) pair scheduled for scoring."""

    passage: Passage
    sentence: Sentence
    e1: Entity
    e2: Entity
    canonical_pair_key: frozenset  # frozenset of canonical types

    @property
    def general_type(self) -> str:
        """The aggregate class name derived from the entity type pair (never NO_RELATION)."""
        # canonical_pair_key is either {'X'} for same-type pairs or {'X','Y'}
        if len(self.canonical_pair_key) == 1:
            (t,) = tuple(self.canonical_pair_key)
            canon = _canonicalize_pair(t, t)
        else:
            a, b = tuple(self.canonical_pair_key)
            canon = _canonicalize_pair(a, b)
        # canon is None only if the pair isn't in the schema; the enumerator filters
        # those out before Pair objects are created, so this is always a real label.
        assert canon is not None
        return canon


def _assign_entities_to_sentences(
    passage_text: str, entities: list[Entity]
) -> list[Sentence]:
    """Split the passage text, then bucket entities into the sentence whose
    span contains their passage-relative offsets. Entities whose offsets
    straddle a sentence boundary are attached to the sentence containing their
    start position (rare in practice; annotations are usually within-sentence)."""
    spans = split_into_sentences(passage_text)
    sentences = [
        Sentence(
            text=passage_text[s:e],
            passage_start=s,
            passage_end=e,
        )
        for s, e in spans
    ]
    for entity in entities:
        for sent in sentences:
            if entity.passage_start >= sent.passage_start and entity.passage_start < sent.passage_end:
                sent.entities.append(entity)
                break
    return sentences


def _dedup_by_concept(entities: list[Entity]) -> list[Entity]:
    """Keep the first mention (lowest offset) per dedup key within a sentence."""
    seen: set[str] = set()
    ordered = sorted(entities, key=lambda e: (e.passage_start, e.passage_end))
    kept: list[Entity] = []
    for e in ordered:
        if e.dedup_key in seen:
            continue
        seen.add(e.dedup_key)
        kept.append(e)
    return kept


def _canonical_pair_key(e1: Entity, e2: Entity) -> frozenset | None:
    """Return the frozenset used to look up the aggregate class, or None if
    the (type1, type2) combination is out of taxonomy (e.g. DISEASE-DISEASE)."""
    if e1.canonical_type == e2.canonical_type:
        key = frozenset([e1.canonical_type])
    else:
        key = frozenset([e1.canonical_type, e2.canonical_type])
    # Reuse the aggregate module's canonical mapping to verify eligibility.
    a, b = (
        (e1.canonical_type, e2.canonical_type)
        if e1.canonical_type != e2.canonical_type
        else (e1.canonical_type, e1.canonical_type)
    )
    if _canonicalize_pair(a, b) is None:
        return None
    return key


def enumerate_pairs(passage: Passage) -> list[Pair]:
    """Populate ``passage.sentences`` (in place) and return every ordered
    intra-sentence pair whose (type1, type2) fits one of the 5 aggregate
    classes. Pairs where both mentions collapse to the same concept via
    dedup are naturally excluded (only one representative survives)."""
    passage.sentences = _assign_entities_to_sentences(passage.text, passage.entities)
    pairs: list[Pair] = []
    for sent in passage.sentences:
        if len(sent.entities) < 2:
            continue
        deduped = _dedup_by_concept(sent.entities)
        sent.entities = deduped
        if len(deduped) < 2:
            continue
        for e1, e2 in product(deduped, repeat=2):
            if e1 is e2:
                continue
            key = _canonical_pair_key(e1, e2)
            if key is None:
                continue
            pairs.append(
                Pair(
                    passage=passage,
                    sentence=sent,
                    e1=e1,
                    e2=e2,
                    canonical_pair_key=key,
                )
            )
    return pairs

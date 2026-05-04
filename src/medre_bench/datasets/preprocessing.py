"""Shared preprocessing utilities: sentence splitting, negative sampling, resampling."""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict

from medre_bench.datasets.base import RelationExample

NO_RELATION = "NO_RELATION"

_SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")


def split_into_sentences(text: str) -> list[tuple[int, int]]:
    """Return list of (start, end) character offsets for each sentence."""
    if not text:
        return []
    spans = []
    cursor = 0
    for match in _SENTENCE_PATTERN.finditer(text):
        end = match.start()
        if end > cursor and text[cursor:end].strip():
            spans.append((cursor, end))
        cursor = match.end()
    if cursor < len(text) and text[cursor:].strip():
        spans.append((cursor, len(text)))
    if not spans:
        spans.append((0, len(text)))
    return spans


def to_sentence_level_examples(
    text: str,
    entities: list[dict],
    relation_pairs: dict[tuple[str, str], str],
    label_to_id: dict[str, int],
    no_relation_label: str,
    doc_id: str = "",
    max_pairs_per_sentence: int = 50,
    seed: int = 42,
) -> list[RelationExample]:
    """Convert document-level annotations into sentence-level RelationExamples.

    For each sentence:
      - Identify entities fully contained in the sentence span.
      - For each ordered entity pair, emit a positive example if a relation
        exists between them, otherwise emit a `no_relation_label` example.
      - Cap pairs per sentence to avoid combinatorial explosion.

    Entity offsets are rebased to be relative to the sentence start.
    """
    sentences = split_into_sentences(text)
    rng = random.Random(seed + hash(doc_id) % (2**32))
    examples: list[RelationExample] = []

    for sent_start, sent_end in sentences:
        sent_text = text[sent_start:sent_end]
        in_sentence = [
            e for e in entities
            if e["start"] >= sent_start and e["end"] <= sent_end
        ]
        if len(in_sentence) < 2:
            continue

        candidate_pairs = []
        for i, e1 in enumerate(in_sentence):
            for j, e2 in enumerate(in_sentence):
                if i == j:
                    continue
                candidate_pairs.append((e1, e2))

        # Always keep positive pairs; subsample negatives if needed
        positives = [(e1, e2) for e1, e2 in candidate_pairs
                     if (e1["id"], e2["id"]) in relation_pairs]
        negatives = [(e1, e2) for e1, e2 in candidate_pairs
                     if (e1["id"], e2["id"]) not in relation_pairs]

        budget = max(max_pairs_per_sentence - len(positives), 0)
        if len(negatives) > budget:
            negatives = rng.sample(negatives, budget)

        for e1, e2 in positives + negatives:
            pair_key = (e1["id"], e2["id"])
            label = relation_pairs.get(pair_key, no_relation_label)
            if label not in label_to_id:
                continue

            examples.append(
                RelationExample(
                    text=sent_text,
                    entity1=e1["text"],
                    entity1_type=e1["type"],
                    entity1_start=e1["start"] - sent_start,
                    entity1_end=e1["end"] - sent_start,
                    entity2=e2["text"],
                    entity2_type=e2["type"],
                    entity2_start=e2["start"] - sent_start,
                    entity2_end=e2["end"] - sent_start,
                    label=label,
                    label_id=label_to_id[label],
                    metadata={"doc_id": doc_id},
                )
            )

    return examples


def process_bigbio_kb_doc(
    doc: dict,
    label_to_id: dict[str, int],
    no_relation_label: str,
    label_remap: dict[str, str] | None = None,
    max_pairs_per_sentence: int = 50,
    seed: int = 42,
) -> list[RelationExample]:
    """Convert a BigBio KB schema document to sentence-level RelationExamples.

    Handles standard BigBio KB layout: passages, entities, relations.
    """
    text = " ".join([p["text"][0] for p in doc["passages"]])

    entities = []
    for entity in doc["entities"]:
        if not entity.get("offsets") or not entity.get("text"):
            continue
        entities.append({
            "id": entity["id"],
            "text": entity["text"][0],
            "type": entity["type"],
            "start": entity["offsets"][0][0],
            "end": entity["offsets"][0][1],
        })

    relation_pairs: dict[tuple[str, str], str] = {}
    for relation in doc["relations"]:
        rel_type = relation["type"]
        if label_remap and rel_type in label_remap:
            rel_type = label_remap[rel_type]
        if rel_type not in label_to_id:
            continue
        relation_pairs[(relation["arg1_id"], relation["arg2_id"])] = rel_type

    return to_sentence_level_examples(
        text=text,
        entities=entities,
        relation_pairs=relation_pairs,
        label_to_id=label_to_id,
        no_relation_label=no_relation_label,
        doc_id=doc.get("id", ""),
        max_pairs_per_sentence=max_pairs_per_sentence,
        seed=seed,
    )


def random_oversample(
    examples: list[RelationExample],
    seed: int = 42,
) -> list[RelationExample]:
    """Balance class distribution via random oversampling with replacement.

    All minority classes are upsampled to the size of the majority class.
    Output is shuffled.
    """
    if not examples:
        return []

    by_label: dict[int, list[RelationExample]] = defaultdict(list)
    for ex in examples:
        by_label[ex.label_id].append(ex)

    target_size = max(len(v) for v in by_label.values())
    rng = random.Random(seed)

    balanced: list[RelationExample] = []
    for label_id, exs in by_label.items():
        balanced.extend(exs)
        if len(exs) < target_size:
            balanced.extend(rng.choices(exs, k=target_size - len(exs)))

    rng.shuffle(balanced)
    return balanced


def compute_class_weights(
    examples: list[RelationExample],
    num_labels: int,
) -> list[float]:
    """Compute inverse-frequency class weights (Sklearn 'balanced' style)."""
    counts = Counter(ex.label_id for ex in examples)
    total = sum(counts.values())
    weights = []
    for i in range(num_labels):
        c = counts.get(i, 0)
        if c == 0:
            weights.append(1.0)
        else:
            weights.append(total / (num_labels * c))
    return weights

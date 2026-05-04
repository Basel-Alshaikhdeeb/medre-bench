"""Shared preprocessing utilities: sentence splitting, negative sampling, resampling."""

from __future__ import annotations

import hashlib
import logging
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from medre_bench.datasets.base import RelationExample, apply_entity_markers

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

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


def _build_marked_texts(examples: list[RelationExample], strategy: str) -> list[str]:
    texts = []
    for ex in examples:
        has_entities = (ex.entity1_start != ex.entity1_end) or (ex.entity2_start != ex.entity2_end)
        if has_entities and strategy != "none":
            texts.append(apply_entity_markers(
                text=ex.text,
                e1_start=ex.entity1_start,
                e1_end=ex.entity1_end,
                e1_type=ex.entity1_type,
                e2_start=ex.entity2_start,
                e2_end=ex.entity2_end,
                e2_type=ex.entity2_type,
                strategy=strategy,
            ))
        else:
            texts.append(ex.text)
    return texts


def _embed_texts(
    texts: list[str],
    embedding_model: str,
    cache_path: Path | None,
    batch_size: int,
) -> "np.ndarray":
    import numpy as np

    if cache_path and cache_path.exists():
        # Cache key includes content hash to invalidate on data change
        cached = np.load(cache_path, allow_pickle=False)
        if cached["text_hash"].item() == _hash_texts(texts):
            logger.info(f"Loaded cached embeddings from {cache_path}")
            return cached["embeddings"]

    from sentence_transformers import SentenceTransformer
    import torch

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    logger.info(f"Embedding {len(texts):,} texts with {embedding_model} on {device}")
    model = SentenceTransformer(embedding_model, device=device)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, embeddings=embeddings, text_hash=np.array(_hash_texts(texts)))
        logger.info(f"Cached embeddings to {cache_path}")

    return embeddings


def _hash_texts(texts: list[str]) -> str:
    h = hashlib.sha1()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def clean_with_tomek(
    examples: list[RelationExample],
    entity_marker_strategy: str = "typed_entity_marker_punct",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    cache_dir: Path | str | None = None,
    cache_key: str | None = None,
    batch_size: int = 128,
) -> list[RelationExample]:
    """Remove Tomek-link majority-class samples to clean class boundaries.

    For each pair of mutual nearest neighbors with different labels, remove
    the member belonging to the more-frequent class (globally). Operates on
    sentence-transformers embeddings of the marked text the model will see.

    Args:
        examples: list of pair-level RelationExamples (typically the train split).
        entity_marker_strategy: how to mark entities before embedding.
        embedding_model: sentence-transformers model id.
        cache_dir: optional directory to cache embeddings between runs.
        cache_key: cache filename stem (e.g. "bc5cdr_train"); skipped if None.
        batch_size: embedding batch size.

    Returns:
        Filtered list of examples with Tomek-link majority-class members removed.
    """
    if not examples:
        return []

    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    cache_path: Path | None = None
    if cache_dir and cache_key:
        safe_model = embedding_model.replace("/", "_")
        cache_path = Path(cache_dir) / f"{cache_key}__{safe_model}.npz"

    texts = _build_marked_texts(examples, entity_marker_strategy)
    embeddings = _embed_texts(texts, embedding_model, cache_path, batch_size)

    nn = NearestNeighbors(n_neighbors=2, metric="cosine", n_jobs=-1)
    nn.fit(embeddings)
    _, neighbor_indices = nn.kneighbors(embeddings)
    nearest = neighbor_indices[:, 1]

    labels = np.array([ex.label_id for ex in examples])
    counts = Counter(labels.tolist())

    tomek_mask = np.zeros(len(examples), dtype=bool)
    for i, j in enumerate(nearest):
        if nearest[j] == i and labels[i] != labels[j]:
            # Remove whichever side belongs to the more-frequent class
            if counts[int(labels[i])] >= counts[int(labels[j])]:
                tomek_mask[i] = True
            else:
                tomek_mask[j] = True

    n_removed = int(tomek_mask.sum())
    keep = ~tomek_mask
    cleaned = [ex for ex, k in zip(examples, keep) if k]

    removed_by_class = Counter(int(labels[i]) for i in np.where(tomek_mask)[0])
    logger.info(
        f"Tomek cleaning: removed {n_removed:,}/{len(examples):,} "
        f"({n_removed/len(examples)*100:.1f}%); per-class removed: {dict(removed_by_class)}"
    )

    return cleaned


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

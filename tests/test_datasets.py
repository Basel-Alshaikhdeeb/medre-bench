"""Tests for dataset base classes and entity markers."""

import pytest

from medre_bench.datasets.base import RelationExample, apply_entity_markers


def test_relation_example_creation():
    ex = RelationExample(
        text="Drug A interacts with Drug B.",
        entity1="Drug A",
        entity1_type="CHEMICAL",
        entity1_start=0,
        entity1_end=6,
        entity2="Drug B",
        entity2_type="CHEMICAL",
        entity2_start=23,
        entity2_end=29,
        label="effect",
        label_id=0,
    )
    assert ex.entity1 == "Drug A"
    assert ex.label_id == 0
    assert ex.metadata == {}


def test_entity_markers_typed_punct():
    text = "Aspirin inhibits COX-2 in cells."
    marked = apply_entity_markers(
        text=text,
        e1_start=0,
        e1_end=7,
        e1_type="CHEMICAL",
        e2_start=17,
        e2_end=22,
        e2_type="PROTEIN",
        strategy="typed_entity_marker_punct",
    )
    assert "@ E1-chemical @" in marked
    assert "@ /E1-chemical @" in marked
    assert "# E2-protein #" in marked
    assert "# /E2-protein #" in marked
    assert "Aspirin" in marked
    assert "COX-2" in marked


def test_entity_markers_standard():
    text = "Gene X causes Disease Y."
    marked = apply_entity_markers(
        text=text,
        e1_start=0,
        e1_end=6,
        e1_type="GENE",
        e2_start=14,
        e2_end=23,
        e2_type="DISEASE",
        strategy="standard",
    )
    assert "[E1]" in marked
    assert "[/E1]" in marked
    assert "[E2]" in marked
    assert "[/E2]" in marked


def test_entity_markers_reversed_order():
    """Test when entity2 appears before entity1 in text."""
    text = "COX-2 is inhibited by Aspirin."
    marked = apply_entity_markers(
        text=text,
        e1_start=22,
        e1_end=29,
        e1_type="CHEMICAL",
        e2_start=0,
        e2_end=5,
        e2_type="PROTEIN",
        strategy="typed_entity_marker_punct",
    )
    # E2 should come first in the text since it appears first
    assert marked.index("E2") < marked.index("E1")


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="Unknown entity marker strategy"):
        apply_entity_markers(
            text="test",
            e1_start=0,
            e1_end=1,
            e1_type="X",
            e2_start=2,
            e2_end=3,
            e2_type="Y",
            strategy="nonexistent",
        )


def test_dataset_registry_classes_have_required_methods():
    import medre_bench.datasets  # noqa: F401
    from medre_bench.registry import DATASET_REGISTRY

    for key in DATASET_REGISTRY.list_available():
        cls = DATASET_REGISTRY.get(key)
        instance = cls()
        assert isinstance(instance.name(), str)
        assert isinstance(instance.num_labels(), int)
        assert instance.num_labels() > 0
        assert isinstance(instance.label_names(), list)
        assert len(instance.label_names()) == instance.num_labels()

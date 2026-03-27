"""Tests for model base classes and registry."""

import pytest


def test_model_registry_classes_have_required_methods():
    import medre_bench.models  # noqa: F401
    from medre_bench.registry import MODEL_REGISTRY

    for key in MODEL_REGISTRY.list_available():
        cls = MODEL_REGISTRY.get(key)
        instance = cls()
        assert isinstance(instance.pretrained_model_name(), str)
        assert len(instance.pretrained_model_name()) > 0


def test_get_entity_marker_tokens():
    from medre_bench.models.base import get_entity_marker_tokens

    # Punct strategy uses punctuation, no special tokens needed
    tokens = get_entity_marker_tokens("typed_entity_marker_punct")
    assert tokens == []

    # Standard strategy needs special tokens
    tokens = get_entity_marker_tokens("standard")
    assert "[E1]" in tokens
    assert "[/E1]" in tokens
    assert "[E2]" in tokens
    assert "[/E2]" in tokens


def test_unknown_marker_strategy_raises():
    from medre_bench.models.base import get_entity_marker_tokens

    with pytest.raises(ValueError):
        get_entity_marker_tokens("nonexistent")

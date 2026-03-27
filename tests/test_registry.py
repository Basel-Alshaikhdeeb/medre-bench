"""Tests for the registry module."""

import pytest

from medre_bench.registry import Registry


def test_register_and_get():
    reg = Registry("test")

    @reg.register("foo")
    class Foo:
        pass

    assert reg.get("foo") is Foo


def test_duplicate_registration_raises():
    reg = Registry("test")

    @reg.register("bar")
    class Bar:
        pass

    with pytest.raises(ValueError, match="already registered"):

        @reg.register("bar")
        class Bar2:
            pass


def test_get_missing_key_raises():
    reg = Registry("test")

    with pytest.raises(KeyError, match="not found"):
        reg.get("nonexistent")


def test_list_available():
    reg = Registry("test")

    @reg.register("b")
    class B:
        pass

    @reg.register("a")
    class A:
        pass

    assert reg.list_available() == ["a", "b"]


def test_model_registry_populated():
    import medre_bench.models  # noqa: F401
    from medre_bench.registry import MODEL_REGISTRY

    available = MODEL_REGISTRY.list_available()
    assert "bert-base" in available
    assert "biobert" in available
    assert "pubmedbert" in available
    assert "scibert" in available
    assert "clinicalbert" in available


def test_dataset_registry_populated():
    import medre_bench.datasets  # noqa: F401
    from medre_bench.registry import DATASET_REGISTRY

    available = DATASET_REGISTRY.list_available()
    assert "chemprot" in available
    assert "ddi" in available
    assert "gad" in available
    assert "biored" in available
    assert "drugprot" in available

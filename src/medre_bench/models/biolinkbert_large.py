"""BioLinkBERT-large model adapter."""

from medre_bench.models.base import BaseREModel
from medre_bench.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("biolinkbert-large")
class BioLinkBERTLargeModel(BaseREModel):
    def pretrained_model_name(self) -> str:
        return "michiyasunaga/BioLinkBERT-large"

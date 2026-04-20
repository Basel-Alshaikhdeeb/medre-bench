"""BioLinkBERT-base model adapter (biomedical RoBERTa variant with link knowledge)."""

from medre_bench.models.base import BaseREModel
from medre_bench.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("biolinkbert-base")
class BioLinkBERTBaseModel(BaseREModel):
    def pretrained_model_name(self) -> str:
        return "michiyasunaga/BioLinkBERT-base"

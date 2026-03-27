"""SciBERT model adapter."""

from medre_bench.models.base import BaseREModel
from medre_bench.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("scibert")
class SciBERTModel(BaseREModel):
    def pretrained_model_name(self) -> str:
        return "allenai/scibert_scivocab_uncased"

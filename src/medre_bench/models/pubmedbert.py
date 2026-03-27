"""PubMedBERT model adapter."""

from medre_bench.models.base import BaseREModel
from medre_bench.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("pubmedbert")
class PubMedBERTModel(BaseREModel):
    def pretrained_model_name(self) -> str:
        return "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"

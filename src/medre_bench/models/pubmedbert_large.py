"""PubMedBERT-large model adapter (biomedical domain, full-text trained)."""

from medre_bench.models.base import BaseREModel
from medre_bench.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("pubmedbert-large")
class PubMedBERTLargeModel(BaseREModel):
    def pretrained_model_name(self) -> str:
        return "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract"

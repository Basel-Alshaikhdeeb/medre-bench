"""PubMed-RoBERTa model adapter (RoBERTa pretrained on PubMed abstracts)."""

from medre_bench.models.base import BaseREModel
from medre_bench.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("pubmed-roberta")
class PubMedRoBERTaModel(BaseREModel):
    def pretrained_model_name(self) -> str:
        return "raynardj/pubmed-roberta-base"

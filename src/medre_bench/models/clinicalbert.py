"""ClinicalBERT model adapter."""

from medre_bench.models.base import BaseREModel
from medre_bench.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("clinicalbert")
class ClinicalBERTModel(BaseREModel):
    def pretrained_model_name(self) -> str:
        return "emilyalsentzer/Bio_ClinicalBERT"

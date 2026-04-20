"""Bio_ClinicalBERT model adapter (clinical notes from MIMIC-III + biomedical text)."""

from medre_bench.models.base import BaseREModel
from medre_bench.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("bio-clinicalbert")
class BioClinicalBERTModel(BaseREModel):
    def pretrained_model_name(self) -> str:
        return "emilyalsentzer/Bio_ClinicalBERT"

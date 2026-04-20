"""BioMed-RoBERTa model adapter (biomedical domain, trained on S2ORC)."""

from medre_bench.models.base import BaseREModel
from medre_bench.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("biomed-roberta")
class BioMedRoBERTaModel(BaseREModel):
    def pretrained_model_name(self) -> str:
        return "allenai/biomed_roberta_base"

"""BioBERT model adapter."""

from medre_bench.models.base import BaseREModel
from medre_bench.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("biobert")
class BioBERTModel(BaseREModel):
    def pretrained_model_name(self) -> str:
        return "dmis-lab/biobert-v1.1"

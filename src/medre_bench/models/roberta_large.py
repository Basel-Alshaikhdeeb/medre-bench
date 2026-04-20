"""RoBERTa-large model adapter (general-domain baseline)."""

from medre_bench.models.base import BaseREModel
from medre_bench.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("roberta-large")
class RoBERTaLargeModel(BaseREModel):
    def pretrained_model_name(self) -> str:
        return "roberta-large"

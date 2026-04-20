"""RoBERTa base model adapter (general-domain baseline)."""

from medre_bench.models.base import BaseREModel
from medre_bench.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("roberta-base")
class RoBERTaBaseModel(BaseREModel):
    def pretrained_model_name(self) -> str:
        return "roberta-base"

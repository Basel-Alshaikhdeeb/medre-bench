"""GatorTron-Medium model adapter."""

from medre_bench.models.base import BaseREModel
from medre_bench.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("gatortron-medium")
class GatorTronMediumModel(BaseREModel):
    def pretrained_model_name(self) -> str:
        return "UFNLP/gatortron-medium"

"""GatorTron model adapter."""

from medre_bench.models.base import BaseREModel
from medre_bench.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("gatortron")
class GatorTronModel(BaseREModel):
    def pretrained_model_name(self) -> str:
        return "UFNLP/gatortron-base"

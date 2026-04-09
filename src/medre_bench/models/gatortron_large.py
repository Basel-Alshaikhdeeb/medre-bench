"""GatorTron-Large model adapter."""

from medre_bench.models.base import BaseREModel
from medre_bench.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("gatortron-large")
class GatorTronLargeModel(BaseREModel):
    def pretrained_model_name(self) -> str:
        return "UFNLP/gatortron-large"

"""BioMedLM model adapter for relation extraction."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from medre_bench.models.base import BaseREModel
from medre_bench.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("biomedlm")
class BioMedLMModel(BaseREModel):
    def pretrained_model_name(self) -> str:
        return "stanford-crfm/BioMedLM"

    def build(
        self,
        num_labels: int,
        entity_marker_tokens: list[str] | None = None,
        dropout: float = 0.1,
    ) -> None:
        model_name = self.pretrained_model_name()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=config)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            config.pad_token_id = self.tokenizer.eos_token_id

        if entity_marker_tokens:
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": entity_marker_tokens}
            )
            self.encoder.resize_token_embeddings(len(self.tokenizer))

        hidden_size = config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass using last-token pooling for causal decoder models."""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = input_ids.shape[0]
        last_token_output = outputs.last_hidden_state[
            torch.arange(batch_size, device=input_ids.device), sequence_lengths
        ]

        last_token_output = self.dropout(last_token_output)
        logits = self.classifier(last_token_output)

        result = {"logits": logits}

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            result["loss"] = loss_fn(logits, labels)

        return result

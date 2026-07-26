"""Abstract base model for relation extraction."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer


class BaseREModel(ABC, nn.Module):
    """Base class for all relation extraction models.

    Subclasses only need to implement `pretrained_model_name()`.
    The base class handles encoder loading, tokenizer setup,
    special token addition, and classification head.
    """

    def __init__(self):
        super().__init__()
        self.tokenizer = None
        self.encoder = None
        self.classifier = None
        self.dropout = None

    @abstractmethod
    def pretrained_model_name(self) -> str:
        """Return the HuggingFace model identifier."""
        ...

    def build(
        self,
        num_labels: int,
        entity_marker_tokens: list[str] | None = None,
        dropout: float = 0.1,
    ) -> None:
        """Initialize encoder, tokenizer, and classification head."""
        model_name = self.pretrained_model_name()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=config)

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
        """Forward pass using CLS token pooling.

        Returns dict with 'logits' and optionally 'loss'.
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)

        result = {"logits": logits}

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            result["loss"] = loss_fn(logits, labels)

        return result


def get_entity_marker_tokens(strategy: str) -> list[str]:
    """Return the special tokens needed for the given entity marker strategy."""
    if strategy == "typed_entity_marker_punct":
        # Uses punctuation (@, #) so no special tokens needed
        return []
    elif strategy == "typed_entity_marker":
        # These are generic; dataset-specific types get added dynamically
        return []
    elif strategy == "standard":
        return ["[E1]", "[/E1]", "[E2]", "[/E2]"]
    else:
        raise ValueError(f"Unknown entity marker strategy: {strategy}")


def align_encoder_vocab_to_state_dict(model: "nn.Module", state_dict: dict) -> None:
    """Resize the wrapped encoder's token embeddings to match the checkpoint's
    vocab size before ``load_state_dict``.

    Loading fails with a size-mismatch error when the pretrained model on HF Hub
    has a different vocab size than what was used during training - either the
    training run added tokens the reconstruction doesn't know about, or HF Hub
    swapped the tokenizer since training. Rather than hard-coding the delta per
    model, we detect the checkpoint's vocab from its word_embeddings tensor and
    resize the encoder via ``resize_token_embeddings``.

    Safe to call even when sizes already match (it's a no-op then). Works
    across BERT, RoBERTa, DistilBERT, GPT-2 and BigBird encoders by scanning
    for the standard token-embedding key names.
    """
    encoder = getattr(getattr(model, "base_model", None), "encoder", None)
    if encoder is None:
        return
    try:
        curr_size = encoder.get_input_embeddings().weight.shape[0]
    except AttributeError:
        return

    embed_suffixes = (
        "word_embeddings.weight",  # BERT / RoBERTa / DistilBERT
        "wte.weight",              # GPT-2
        "embed_tokens.weight",     # BigBird / some seq2seq
    )
    for key, tensor in state_dict.items():
        if key.endswith(embed_suffixes):
            ckpt_size = tensor.shape[0]
            if ckpt_size != curr_size:
                encoder.resize_token_embeddings(ckpt_size)
            return

from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import RobertaConfig

from model import Easyrec


def _make_model(do_mlm: bool = False) -> Easyrec:
    config = RobertaConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=16,
    )
    args = SimpleNamespace(
        pooler_type="cls",
        temp=0.05,
        do_mlm=do_mlm,
        mlm_weight=0.1,
    )
    with patch.object(Easyrec, "init_weights", lambda self: None):
        model = Easyrec(config, model_args=args)
    model.eval()
    return model


def _base_inputs(config: RobertaConfig) -> dict:
    input_ids = torch.tensor([[0, 5, 2, 1], [0, 6, 2, 1]])
    attention_mask = input_ids.ne(config.pad_token_id)
    return {
        "user_input_ids": input_ids,
        "user_attention_mask": attention_mask,
        "pos_item_input_ids": input_ids,
        "pos_item_attention_mask": attention_mask,
        "neg_item_input_ids": input_ids,
        "neg_item_attention_mask": attention_mask,
    }


def _count_encoder_calls(model: Easyrec) -> list[int]:
    original_forward = model.roberta.forward
    calls = [0]

    def counted_forward(*args, **kwargs):
        calls[0] += 1
        return original_forward(*args, **kwargs)

    model.roberta.forward = counted_forward
    return calls


def test_forward_without_optional_mlm_inputs():
    model = _make_model()
    outputs = model(**_base_inputs(model.config))

    assert outputs.loss is not None
    assert torch.isfinite(outputs.loss)


def test_disabled_mlm_skips_encoder_pass_even_when_batch_has_mlm_inputs():
    model = _make_model(do_mlm=False)
    calls = _count_encoder_calls(model)
    mlm_input_ids = torch.tensor([[0, 4, 2, 1]] * 6)
    mlm_labels = torch.full_like(mlm_input_ids, -100)
    mlm_labels[0, 1] = 5

    outputs = model(
        **_base_inputs(model.config),
        mlm_input_ids=mlm_input_ids,
        mlm_attention_mask=mlm_input_ids.ne(model.config.pad_token_id),
        mlm_labels=mlm_labels,
    )

    assert calls[0] == 3
    assert torch.isfinite(outputs.loss)


def test_enabled_mlm_keeps_auxiliary_encoder_pass():
    model = _make_model(do_mlm=True)
    calls = _count_encoder_calls(model)
    mlm_input_ids = torch.tensor([[0, 4, 2, 1]] * 6)
    mlm_labels = torch.full_like(mlm_input_ids, -100)
    mlm_labels[0, 1] = 5

    outputs = model(
        **_base_inputs(model.config),
        mlm_input_ids=mlm_input_ids,
        mlm_attention_mask=mlm_input_ids.ne(model.config.pad_token_id),
        mlm_labels=mlm_labels,
    )

    assert calls[0] == 4
    assert torch.isfinite(outputs.loss)

from unittest.mock import MagicMock

import numpy as np


def _fake_session(hidden_size: int = 4) -> MagicMock:
    session = MagicMock()
    input_ids = MagicMock()
    input_ids.name = "input_ids"
    attention_mask = MagicMock()
    attention_mask.name = "attention_mask"
    session.get_inputs.return_value = [input_ids, attention_mask]
    session.run.return_value = [np.ones((1, 3, hidden_size), dtype=np.float32)]
    return session


def _fake_tokenizer() -> MagicMock:
    tokenizer = MagicMock()
    encoded = MagicMock()
    encoded.ids = [1, 2, 3]
    encoded.attention_mask = [1, 1, 0]
    tokenizer.encode.return_value = encoded
    return tokenizer


def test_onnx_encoder_returns_normalized_vector() -> None:
    from embed_cv_search import OnnxEncoder

    encoder = OnnxEncoder(_fake_session(), _fake_tokenizer())
    vector = encoder.encode("développeur python")

    assert vector.shape == (4,)
    assert np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5)


def test_onnx_encoder_passes_token_type_ids_when_expected() -> None:
    from embed_cv_search import OnnxEncoder

    session = _fake_session()
    token_type_ids = MagicMock()
    token_type_ids.name = "token_type_ids"
    session.get_inputs.return_value = [*session.get_inputs.return_value, token_type_ids]

    encoder = OnnxEncoder(session, _fake_tokenizer())
    encoder.encode("développeur python")

    feed = session.run.call_args.args[1]
    assert "token_type_ids" in feed

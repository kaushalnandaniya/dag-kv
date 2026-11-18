import pytest
import torch
from transformers import (
    Gemma2Config, Gemma2ForCausalLM,
    LlamaConfig, LlamaForCausalLM,
    MistralConfig, MistralForCausalLM,
    Qwen2Config, Qwen2ForCausalLM,
)

from dag_kv import (
    AutoDAGKVEngine,
    DAGKVModelWrapper,
    Gemma2DAGKVEngine,
    LlamaDAGKVEngine,
    MistralDAGKVEngine,
    Qwen2DAGKVEngine,
)


class _DummyTokenizer:
    eos_token_id = 9999

    def encode(self, text, add_special_tokens=False):
        return [max(3, ord(c) % 250) for c in text]

    def decode(self, ids, skip_special_tokens=True):
        if isinstance(ids, int):
            ids = [ids]
        return "".join(chr(65 + (i % 26)) for i in ids)


def test_auto_engine_routing():
    tok = _DummyTokenizer()

    # Llama routing
    llama_cfg = LlamaConfig(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2)
    llama_model = LlamaForCausalLM(llama_cfg).eval()
    engine_llama = AutoDAGKVEngine.from_model(llama_model, tok, max_num_blocks=16, block_size=16)
    assert isinstance(engine_llama, LlamaDAGKVEngine)

    # Mistral routing
    mistral_cfg = MistralConfig(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2)
    mistral_model = MistralForCausalLM(mistral_cfg).eval()
    engine_mistral = AutoDAGKVEngine.from_model(mistral_model, tok, max_num_blocks=16, block_size=16)
    assert isinstance(engine_mistral, MistralDAGKVEngine)

    # Gemma routing
    gemma_cfg = Gemma2Config(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2, head_dim=32)
    gemma_model = Gemma2ForCausalLM(gemma_cfg).eval()
    engine_gemma = AutoDAGKVEngine.from_model(gemma_model, tok, max_num_blocks=16, block_size=16)
    assert isinstance(engine_gemma, Gemma2DAGKVEngine)

    # Qwen2 routing
    qwen_cfg = Qwen2Config(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2)
    qwen_model = Qwen2ForCausalLM(qwen_cfg).eval()
    engine_qwen = AutoDAGKVEngine.from_model(qwen_model, tok, max_num_blocks=16, block_size=16)
    assert isinstance(engine_qwen, Qwen2DAGKVEngine)


def test_llama_end_to_end_dag():
    torch.manual_seed(42)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev.startswith("cuda") else torch.float32

    cfg = LlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=1024,
    )
    model = LlamaForCausalLM(cfg).to(device=dev, dtype=dtype).eval()
    tok = _DummyTokenizer()

    wrapper = DAGKVModelWrapper(
        model=model,
        tokenizer=tok,
        max_num_blocks=32,
        block_size=16,
    )
    assert isinstance(wrapper.engine, LlamaDAGKVEngine)

    c1 = wrapper.prefill_chunk("Doc A " * 4)
    c2 = wrapper.prefill_chunk("Doc B " * 4)
    assert c1.chunk_id != c2.chunk_id

    out = wrapper.generate([c1, c2], prompt_suffix="Q?", max_new_tokens=4)
    assert len(out) == 4


def test_mistral_end_to_end_dag():
    torch.manual_seed(42)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev.startswith("cuda") else torch.float32

    cfg = MistralConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=1024,
    )
    model = MistralForCausalLM(cfg).to(device=dev, dtype=dtype).eval()
    tok = _DummyTokenizer()

    wrapper = DAGKVModelWrapper(
        model=model,
        tokenizer=tok,
        max_num_blocks=32,
        block_size=16,
    )
    assert isinstance(wrapper.engine, MistralDAGKVEngine)

    c1 = wrapper.prefill_chunk("Mistral chunk alpha " * 3)
    c2 = wrapper.prefill_chunk("Mistral chunk beta " * 3)

    out = wrapper.generate([c1, c2], prompt_suffix="Q?", max_new_tokens=4)
    assert len(out) == 4


def test_gemma2_end_to_end_dag():
    torch.manual_seed(42)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev.startswith("cuda") else torch.float32

    cfg = Gemma2Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=1024,
    )
    model = Gemma2ForCausalLM(cfg).to(device=dev, dtype=dtype).eval()
    tok = _DummyTokenizer()

    wrapper = DAGKVModelWrapper(
        model=model,
        tokenizer=tok,
        max_num_blocks=32,
        block_size=16,
    )
    assert isinstance(wrapper.engine, Gemma2DAGKVEngine)

    c1 = wrapper.prefill_chunk("Gemma text segment 1 " * 3)
    c2 = wrapper.prefill_chunk("Gemma text segment 2 " * 3)

    out = wrapper.generate([c1, c2], prompt_suffix="Q?", max_new_tokens=4)
    assert len(out) == 4

"""Automated Model Architecture Detector and Factory for DAG-KV Engines."""

from typing import Optional, Union
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from .base_engine import DAGKVEngine
from .gemma_engine import Gemma2DAGKVEngine
from .llama_engine import LlamaDAGKVEngine
from .mistral_engine import MistralDAGKVEngine
from .qwen2_dag_patch import Qwen2DAGKVEngine


class AutoDAGKVEngine:
    """Factory creating architecture-optimized DAGKVEngines automatically."""

    @classmethod
    def from_model(
        cls,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        **kwargs,
    ) -> DAGKVEngine:
        """Inspects model configuration and instantiates the matching DAGKVEngine subclass."""
        model_type = getattr(model.config, "model_type", "").lower()
        class_name = model.__class__.__name__.lower()

        if "gemma" in model_type or "gemma" in class_name:
            return Gemma2DAGKVEngine(model=model, tokenizer=tokenizer, **kwargs)
        elif "llama" in model_type or "llama" in class_name:
            return LlamaDAGKVEngine(model=model, tokenizer=tokenizer, **kwargs)
        elif "mistral" in model_type or "mistral" in class_name or "mixtral" in model_type:
            return MistralDAGKVEngine(model=model, tokenizer=tokenizer, **kwargs)
        else:
            return Qwen2DAGKVEngine(model=model, tokenizer=tokenizer, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ) -> DAGKVEngine:
        """Loads model and tokenizer from Hugging Face and returns the appropriate DAGKVEngine."""
        target_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        target_dtype = dtype or (torch.float16 if str(target_device).startswith("cuda") else torch.float32)

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            dtype=target_dtype,
            device_map=target_device if str(target_device) != "cpu" else None,
        )
        if str(target_device) == "cpu":
            model = model.to("cpu")
        model.eval()

        return cls.from_model(
            model=model,
            tokenizer=tokenizer,
            device=target_device,
            dtype=target_dtype,
            **kwargs,
        )

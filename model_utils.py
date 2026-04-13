from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


TORCH_DTYPE_MAP = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}

DEFAULT_POLICY_ADAPTER_NAME = "policy"
DEFAULT_REFERENCE_ADAPTER_NAME = "reference"


def set_peft_base_model_name_or_path(model, base_model_name_or_path: str):
    """Keep PEFT adapter metadata pointed at the local base model path."""
    peft_config = getattr(model, "peft_config", None)
    if peft_config is None:
        return

    for config in peft_config.values():
        config.base_model_name_or_path = base_model_name_or_path


def resolve_torch_dtype(torch_dtype):
    """Resolve a string dtype name to a torch dtype."""
    if torch_dtype is None or isinstance(torch_dtype, torch.dtype):
        return torch_dtype

    normalized = torch_dtype.lower()
    if normalized not in TORCH_DTYPE_MAP:
        supported = ", ".join(sorted(TORCH_DTYPE_MAP))
        raise ValueError(f"Unsupported torch dtype '{torch_dtype}'. Supported values: {supported}")

    return TORCH_DTYPE_MAP[normalized]


def load_tokenizer(model_name_or_path):
    """Load a tokenizer and ensure a pad token is set."""
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_lm(
    base_model_name_or_path,
    adapter_name_or_path=None,
    torch_dtype="bfloat16",
    device_map=None,
    merge_adapter=False,
    use_flash_attention_2=False,
):
    """Load a causal LM, optionally applying and merging a PEFT adapter."""
    resolved_dtype = resolve_torch_dtype(torch_dtype)
    model_kwargs = {
        'torch_dtype': resolved_dtype,
        'device_map': device_map,
        'low_cpu_mem_usage': True,
    }
    if use_flash_attention_2:
        model_kwargs['attn_implementation'] = 'flash_attention_2'

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        **model_kwargs,
    )

    if adapter_name_or_path:
        model = PeftModel.from_pretrained(model, adapter_name_or_path)
        set_peft_base_model_name_or_path(model, base_model_name_or_path)
        if merge_adapter:
            model = model.merge_and_unload()
            if resolved_dtype is not None:
                model = model.to(dtype=resolved_dtype)

    return model


def load_shared_reference_policy_model(
    base_model_name_or_path,
    policy_adapter_name_or_path,
    reference_adapter_name_or_path: Optional[str] = None,
    torch_dtype="bfloat16",
    device_map=None,
    policy_adapter_name: str = DEFAULT_POLICY_ADAPTER_NAME,
    reference_adapter_name: str = DEFAULT_REFERENCE_ADAPTER_NAME,
    use_flash_attention_2: bool = False,
):
    """Load one base model with separate trainable policy and frozen reference adapters."""
    if not policy_adapter_name_or_path:
        raise ValueError("policy_adapter_name_or_path must be provided for shared-base SPIN training.")
    if policy_adapter_name == reference_adapter_name:
        raise ValueError("policy_adapter_name and reference_adapter_name must be different.")

    resolved_dtype = resolve_torch_dtype(torch_dtype)
    model_kwargs = {
        'torch_dtype': resolved_dtype,
        'device_map': device_map,
        'low_cpu_mem_usage': True,
    }
    if use_flash_attention_2:
        model_kwargs['attn_implementation'] = 'flash_attention_2'

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        **model_kwargs,
    )

    model = PeftModel.from_pretrained(
        base_model,
        policy_adapter_name_or_path,
        adapter_name=policy_adapter_name,
        is_trainable=True,
        low_cpu_mem_usage=True,
    )

    reference_source = reference_adapter_name_or_path or policy_adapter_name_or_path
    model.load_adapter(
        reference_source,
        adapter_name=reference_adapter_name,
        is_trainable=False,
        low_cpu_mem_usage=True,
    )

    model.set_adapter(policy_adapter_name)
    model.set_requires_grad(policy_adapter_name, True)
    model.set_requires_grad(reference_adapter_name, False)
    set_peft_base_model_name_or_path(model, base_model_name_or_path)

    return model
import torch


def torch_dtype(dtype: str | torch.dtype) -> torch.dtype:
    """Convert a string to a torch dtype."""
    if isinstance(dtype, torch.dtype):
        return dtype

    if dtype in ["fp16", "float16"]:
        return torch.float16

    if dtype in ["fp32", "float32"]:
        return torch.float32

    if dtype in ["bf16", "bfloat16"]:
        return torch.bfloat16

    try:
        return getattr(torch, dtype)
    except AttributeError:
        raise ValueError(f"Unknown dtype: {dtype}")

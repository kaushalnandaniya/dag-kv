"""Adaptive KV Cache Quantization (INT8 / INT4 / FP16).

Enables 50% to 75% VRAM compression for running large multi-chunk DAG-KV models
on consumer GPUs (e.g. 8GB/12GB RTX 2080/3060/4060) and Apple Silicon MacBooks.
"""

from typing import Optional, Tuple
import torch


class KVQuantizer:
    """Vectorized symmetric quantizer and dequantizer for KV cache tensors."""

    INT4_GROUP_SIZE: int = 32
    INT4_K_GROUP_SIZE = 8
    INT4_V_GROUP_SIZE = 32

    @classmethod
    def num_int4_groups(cls, head_dim: int, group_size: Optional[int] = None) -> int:
        """Returns number of INT4 quantization groups along head_dim."""
        g = cls.INT4_GROUP_SIZE if group_size is None else group_size
        if g > 0 and head_dim % g == 0:
            return head_dim // g
        return 1

    @staticmethod
    def quantize_int8(
        tensor: torch.Tensor,
        dim: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs symmetric INT8 quantization along the specified dimension."""
        orig_dtype = tensor.dtype
        abs_max = torch.amax(torch.abs(tensor), dim=dim, keepdim=True)
        scale = torch.clamp(abs_max / 127.0, min=1e-8)
        quantized = torch.clamp(torch.round(tensor / scale), -128, 127).to(torch.int8)
        return quantized, scale.to(orig_dtype)

    @staticmethod
    def dequantize_int8(
        quantized: torch.Tensor,
        scale: torch.Tensor,
        out_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Dequantizes INT8 tensor back to floating point."""
        dtype = out_dtype or scale.dtype
        return quantized.to(dtype) * scale

    @classmethod
    def quantize_int4(
        cls,
        tensor: torch.Tensor,
        dim: int = -1,
        group_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Group-wise symmetric INT4 quantization packed into uint8 bytes (2 nibbles per byte)."""
        orig_dtype = tensor.dtype
        shape = tensor.shape
        norm_dim = dim if dim >= 0 else len(shape) + dim
        assert norm_dim == len(shape) - 1, "INT4 packing requires dim=-1"
        last_dim = shape[-1]
        assert last_dim % 2 == 0, f"Last dimension {last_dim} must be even for INT4 packing"

        num_groups = cls.num_int4_groups(last_dim, group_size=group_size)
        eff_group = last_dim // num_groups

        xg = tensor.reshape(*shape[:-1], num_groups, eff_group)
        abs_max = torch.amax(torch.abs(xg), dim=-1, keepdim=True)
        scale = torch.clamp(abs_max / 7.0, min=1e-8)
        q_int = torch.clamp(torch.round(xg / scale), -8, 7).to(torch.int32).reshape(shape)
        q_unsigned = (q_int + 8) & 0x0F

        low_nibble = q_unsigned[..., 0::2]
        high_nibble = q_unsigned[..., 1::2]
        packed = (low_nibble | (high_nibble << 4)).to(torch.uint8)

        return packed, scale.squeeze(-1).to(orig_dtype)

    @staticmethod
    def dequantize_int4(
        packed: torch.Tensor,
        scale: torch.Tensor,
        out_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Unpacks and dequantizes group-wise INT4 bytes back to floating point."""
        dtype = out_dtype or scale.dtype
        low = (packed & 0x0F).to(torch.int32) - 8
        high = ((packed >> 4) & 0x0F).to(torch.int32) - 8

        shape = list(packed.shape)
        head_dim = shape[-1] * 2
        shape[-1] = head_dim
        unpacked = torch.empty(shape, dtype=torch.int32, device=packed.device)
        unpacked[..., 0::2] = low
        unpacked[..., 1::2] = high

        num_groups = scale.shape[-1]
        group_size = head_dim // num_groups
        scale_exp = scale.unsqueeze(-1).expand(*shape[:-1], num_groups, group_size).reshape(shape)
        return unpacked.to(dtype) * scale_exp.to(dtype)

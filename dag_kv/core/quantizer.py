"""Adaptive KV Cache Quantization (INT8 / INT4 / FP16).

Enables 50% to 75% VRAM compression for running large multi-chunk DAG-KV models
on consumer GPUs (e.g. 8GB/12GB RTX 2080/3060/4060) and Apple Silicon MacBooks.
"""

from typing import Optional, Tuple
import torch


class KVQuantizer:
    """Vectorized symmetric quantizer and dequantizer for KV cache tensors."""

    @staticmethod
    def quantize_int8(
        tensor: torch.Tensor,
        dim: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs symmetric INT8 quantization along the specified dimension.

        Args:
            tensor: Floating point tensor (FP16, BF16, or FP32)
            dim: Dimension across which scale is computed (default: head_dim)

        Returns:
            (quantized_int8, scale_factors)
        """
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

    @staticmethod
    def quantize_int4(
        tensor: torch.Tensor,
        dim: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Symmetric INT4 quantization packed into uint8 bytes (2 nibbles per byte).

        Yields a 4x (75%) reduction in KV memory footprint.
        """
        orig_dtype = tensor.dtype
        shape = tensor.shape
        last_dim = shape[dim if dim >= 0 else len(shape) + dim]
        assert last_dim % 2 == 0, f"Last dimension {last_dim} must be even for INT4 packing"

        abs_max = torch.amax(torch.abs(tensor), dim=dim, keepdim=True)
        scale = torch.clamp(abs_max / 7.0, min=1e-8)
        q_int = torch.clamp(torch.round(tensor / scale), -8, 7).to(torch.int32)
        # Convert signed [-8, 7] to unsigned 4-bit [0, 15]
        q_unsigned = (q_int + 8) & 0x0F

        # Pack adjacent pairs: low nibble = even index, high nibble = odd index
        low_nibble = q_unsigned[..., 0::2]
        high_nibble = q_unsigned[..., 1::2]
        packed = (low_nibble | (high_nibble << 4)).to(torch.uint8)

        return packed, scale.to(orig_dtype)

    @staticmethod
    def dequantize_int4(
        packed: torch.Tensor,
        scale: torch.Tensor,
        out_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Unpacks and dequantizes INT4 bytes back to floating point."""
        dtype = out_dtype or scale.dtype
        low = (packed & 0x0F).to(torch.int32) - 8
        high = ((packed >> 4) & 0x0F).to(torch.int32) - 8

        # Interleave low and high nibbles
        shape = list(packed.shape)
        shape[-1] = shape[-1] * 2
        unpacked = torch.empty(shape, dtype=torch.int32, device=packed.device)
        unpacked[..., 0::2] = low
        unpacked[..., 1::2] = high

        return unpacked.to(dtype) * scale

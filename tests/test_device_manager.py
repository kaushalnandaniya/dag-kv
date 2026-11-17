"""Tests for Hardware Abstraction Layer & DeviceManager."""

import pytest
import torch

from dag_kv.devices.device_manager import DeviceManager, get_device_manager
from dag_kv.devices.shard_config import QuantizationMode, ShardConfig, TopologyMode


def test_device_manager_singleton_and_probing():
    """Verifies that DeviceManager discovers system devices and respects singleton pattern."""
    mgr1 = get_device_manager()
    mgr2 = DeviceManager()
    assert mgr1 is mgr2

    devices = mgr1.devices
    assert len(devices) >= 1  # CPU must always be present

    cpu_dev = next(d for d in devices if d.device_type == "cpu")
    assert cpu_dev is not None
    assert cpu_dev.total_memory > 0


def test_shard_config_presets():
    """Tests all standard preset configurations."""
    # CPU fallback
    cfg_cpu = ShardConfig.cpu_fallback(block_size=8, max_num_blocks=512)
    assert cfg_cpu.is_cpu
    assert cfg_cpu.dtype == torch.float32
    assert cfg_cpu.block_size == 8

    # Single GPU
    cfg_single = ShardConfig.single_gpu(device="cuda:0", block_size=16)
    assert cfg_single.is_cuda
    assert cfg_single.topology == TopologyMode.SINGLE_DEVICE

    # Dual GPU pipeline
    cfg_pipe = ShardConfig.dual_gpu_pipeline(target_device="cuda:0", prefill_device="cuda:1")
    assert cfg_pipe.topology == TopologyMode.PIPELINE
    assert cfg_pipe.prefill_device == torch.device("cuda:1")
    assert cfg_pipe.device == torch.device("cuda:0")

    # CPU offload
    cfg_offload = ShardConfig.cpu_offloaded_kv(compute_device="cuda:0", block_size=32)
    assert cfg_offload.is_cpu_offloaded
    assert cfg_offload.device == torch.device("cuda:0")
    assert cfg_offload.kv_device == torch.device("cpu")
    assert cfg_offload.block_size == 32


def test_auto_detect_topology():
    """Tests topology auto-detection under various hardware configurations."""
    mgr = get_device_manager()

    # Automatic topology selection
    config = mgr.auto_detect_topology(model_size_bytes=1024 * 1024 * 1024)
    assert isinstance(config, ShardConfig)
    assert config.topology in (TopologyMode.SINGLE_DEVICE, TopologyMode.PIPELINE, TopologyMode.CPU_OFFLOAD)

    # Explicit CPU request
    config_cpu = mgr.auto_detect_topology(preferred_topology="cpu")
    assert config_cpu.is_cpu

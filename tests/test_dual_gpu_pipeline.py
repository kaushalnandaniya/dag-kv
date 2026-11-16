import pytest
import torch
from dag_kv.worker.dual_gpu_pipeline import DualGPUPipeline


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires 2 GPUs")
def test_dual_gpu_disaggregated_prefill_and_stitch():
    pipe = DualGPUPipeline(
        model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
        inference_device="cuda:0",
        prefill_device="cuda:1",
        max_num_blocks=512,
    )
    doc_a = "Document A: Alexander Fleming discovered penicillin at St. Mary's Hospital in London in 1928.\n"
    doc_b = "Document B: Penicillin was mass-produced during World War II, saving millions of soldiers' lives.\n"

    fut_a = pipe.prefill_chunk_async(doc_a)
    fut_b = pipe.prefill_chunk_async(doc_b)
    chunk_a = pipe.register_transferred_chunk(fut_a.result())
    chunk_b = pipe.register_transferred_chunk(fut_b.result())

    assert chunk_a.k_cache.device == torch.device("cuda:0")
    assert chunk_b.k_cache.device == torch.device("cuda:0")

    _, ans = pipe.stitch_and_generate(
        chunks=[chunk_b, chunk_a],
        prompt_suffix="Based on the documents above, answer with ONLY the city name (1 word): In what city did Alexander Fleming discover penicillin?\nAnswer:",
        max_new_tokens=4,
    )
    assert "london" in ans.lower()

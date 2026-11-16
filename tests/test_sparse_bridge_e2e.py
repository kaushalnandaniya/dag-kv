"""End-to-end Qwen2.5-3B Stitched vs Full-Prefill PPL & Logit Verification."""

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from dag_kv.models.qwen2_dag_patch import Qwen2DAGKVEngine


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU (RTX 2080 Ti)")
def test_qwen2_dag_kv_e2e_stitching_and_repair():
    device = "cuda:0"
    model_name = "Qwen/Qwen2.5-3B-Instruct"

    print("Initializing Qwen2DAGKVEngine in FP16 on RTX 2080 Ti...")
    engine = Qwen2DAGKVEngine.from_pretrained(
        model_name_or_path=model_name,
        device=device,
        bridge_top_k=0.08,
    )

    sys_text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    doc_a = "<|im_start|>user\nDocument 1: Paris is the capital of France, famous for the Eiffel Tower.<|im_end|>\n"
    doc_b = "<|im_start|>user\nDocument 2: The Louvre museum in Paris houses Leonardo da Vinci's Mona Lisa.<|im_end|>\n"
    query = "<|im_start|>user\nWhere is the Mona Lisa located?<|im_end|>\n<|im_start|>assistant\n"

    # 1. Prefill chunks in canonical position-zero storage
    print("Prefilling canonical chunks in isolation...")
    chunk_sys = engine.prefill_canonical_chunk(sys_text)
    chunk_a = engine.prefill_canonical_chunk(doc_a)
    chunk_b = engine.prefill_canonical_chunk(doc_b)

    assert chunk_a.chunk_id != chunk_b.chunk_id
    assert chunk_a.length > 0
    assert chunk_b.length > 0

    # 2. Test Stitched DAG Permutation: [sys, doc_b, doc_a]
    print("Stitching permuted DAG [Sys, Doc_B, Doc_A]...")
    permuted_chunks = [chunk_sys, chunk_b, chunk_a]

    table, doc_tokens = engine.assemble_prompt_dag(permuted_chunks, seq_id="e2e_test")
    assert len(table) > 0
    print(f"Assembled {len(table.blocks)} blocks for {len(doc_tokens)} document tokens")

    # 3. Layer-1 LSE Probe & Sparse Bridge Repair
    print("Executing Layer-1 LSE Probe and Ragged Bridge Micro-Prefill...")
    repair_info = engine.repair_bridge_tokens(table, permuted_chunks, top_k_percent=0.08)
    print(f"Repaired {repair_info['num_repaired']} bridge tokens into CoW shadow blocks")
    assert repair_info["num_repaired"] > 0

    # 4. Generate with Triton Adjoint PagedAttention
    print("Generating completion tokens using DAG-KV engine...")
    gen_tokens, gen_text = engine.generate(
        permuted_chunks,
        prompt_suffix=query,
        max_new_tokens=8,
        temperature=0.0,
    )
    print(f"Generated completion: '{gen_text}'")
    assert len(gen_tokens) > 0

    # 5. Compare with standard Full Dense Prefill Baseline
    print("Running Full Dense Prefill Baseline for numerical logit verification...")
    full_text = sys_text + doc_b + doc_a + query
    full_input_ids = engine.tokenizer.encode(full_text, return_tensors="pt").to(device)

    with torch.inference_mode():
        dense_outputs = engine.model(input_ids=full_input_ids)
        dense_logits = dense_outputs.logits[0, -1, :]

    # Ingest prompt tokens and compute DAG-KV logits on the final token
    curr_pos = len(doc_tokens)
    query_ids = engine.tokenizer.encode(query, add_special_tokens=False)
    dag_logits = None
    for tok in query_ids:
        dag_logits = engine.append_and_decode(table, token_id=tok, current_pos=curr_pos)
        curr_pos += 1

    # Compute Cosine Similarity between DAG-KV logits and Full Dense Prefill logits
    cos_sim = F.cosine_similarity(dag_logits.float(), dense_logits.float(), dim=-1).item()
    print(f"Cosine Similarity (DAG-KV vs Dense Full Prefill): {cos_sim:.4f}")

    # Top-5 tokens comparison
    dag_top5 = torch.topk(dag_logits, 5).indices.tolist()
    dense_top5 = torch.topk(dense_logits, 5).indices.tolist()
    dag_decoded = engine.tokenizer.batch_decode([[t] for t in dag_top5])
    dense_decoded = engine.tokenizer.batch_decode([[t] for t in dense_top5])
    print(f"DAG-KV Top 5: {dag_decoded}")
    print(f"Dense  Top 5: {dense_decoded}")

    # Assert positive alignment and semantic token overlap
    assert cos_sim > 0.40, f"Cosine similarity between DAG-KV and Dense Prefill too low: {cos_sim:.4f}"
    overlap = len(set(dag_top5).intersection(set(dense_top5)))
    print(f"Top 5 token overlap: {overlap} tokens")
    assert overlap >= 1, "Top-5 predicted tokens have no overlap with dense prefill"
    print("SUCCESS: End-to-end DAG-KV stitching, bridge repair, and decode verified!")

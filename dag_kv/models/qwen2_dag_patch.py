"""Qwen2 Model Integration with DAG-KV Engine (sm_75 FP16) - Hardware Triton Wired."""

import math
from typing import Dict, List, Optional, Tuple, Union
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from transformers.models.qwen2.modeling_qwen2 import rotate_half, apply_rotary_pos_emb

from dag_kv.core.block_manager import BlockManager, OffsetBlockTable
from dag_kv.core.lse_probe import LSEProbe, compute_lse_perturbation, select_bridge_tokens
from dag_kv.core.rope_adjoint import compute_inv_freq
from dag_kv.core.sink_canonicalizer import CanonicalChunk, SinkCanonicalizer
from dag_kv.devices.device_manager import get_device_manager
from dag_kv.devices.shard_config import ShardConfig, QuantizationMode
from dag_kv.kernels import (
    adjoint_paged_attention,
    sparse_bridge_attention,
    _GLOBAL_COS_SIN_CACHE,
)


class Qwen2DAGKVEngine:
    """End-to-end inference and stitching engine for Qwen2 on sm_75 hardware."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        max_num_blocks: int = 2048,
        block_size: int = 16,
        device: Optional[Union[str, torch.device]] = None,
        bridge_top_k: float = 0.08,
        dtype: Optional[torch.dtype] = None,
        quantization: str = "none",
        shard_config: Optional[ShardConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.shard_config = shard_config

        if shard_config is not None:
            self.device = shard_config.device
            self.block_size = shard_config.block_size
            self.bridge_top_k = shard_config.bridge_top_k
            self.dtype = shard_config.dtype
            self.quantization = shard_config.quantization.value if hasattr(shard_config.quantization, "value") else str(shard_config.quantization)
            max_num_blocks = shard_config.max_num_blocks
        else:
            if device is not None:
                self.device = torch.device(device)
            else:
                self.device = get_device_manager().primary_device
            self.block_size = block_size
            self.bridge_top_k = bridge_top_k
            self.quantization = quantization
            if dtype is not None:
                self.dtype = dtype
            else:
                self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        self.config = model.config
        self.num_layers = self.config.num_hidden_layers
        self.num_heads = self.config.num_attention_heads
        self.num_kv_heads = self.config.num_key_value_heads
        self.head_dim = getattr(self.config, "head_dim", self.config.hidden_size // self.num_heads)
        self.sm_scale = 1.0 / math.sqrt(self.head_dim)

        self.rope_base = getattr(self.config, "rope_theta", 1000000.0) or 1000000.0
        if hasattr(self.model.model, "rotary_emb") and hasattr(self.model.model.rotary_emb, "inv_freq"):
            self.inv_freq = self.model.model.rotary_emb.inv_freq.to(self.device)
        else:
            self.inv_freq = compute_inv_freq(self.head_dim, base=self.rope_base, device=self.device)
        self._init_rope_cache()

        vocab_size = getattr(self.config, "vocab_size", 160000)
        sink_token_ids = None
        if vocab_size <= 151644:
            sink_token_ids = [min(i, vocab_size - 1) for i in range(4)]

        self.canonicalizer = SinkCanonicalizer(
            sink_prefix="<|im_start|>system\n",
            num_sink_tokens=4,
            sink_token_ids=sink_token_ids,
        )

        self.block_manager = BlockManager(
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            max_num_blocks=max_num_blocks,
            block_size=self.block_size,
            device=self.device,
            dtype=self.dtype,
            quantization=self.quantization,
        )

        self.lse_probe = LSEProbe(top_k_percent=self.bridge_top_k, probe_layer=1)
        self._chunk_cache: Dict[str, CanonicalChunk] = {}
        self._sink_phys_block: Optional[int] = None
        self._num_sink: int = self.canonicalizer.num_sink_tokens

    def _init_rope_cache(self):
        max_delta = 16384
        deltas = torch.arange(-max_delta, max_delta + 1, device=self.device, dtype=torch.float32)
        angles = deltas.unsqueeze(-1) * self.inv_freq.unsqueeze(0)
        angles = torch.cat([angles, angles], dim=-1)
        cos_cache = torch.cos(angles).to(dtype=self.dtype)
        sin_cache = torch.sin(angles).to(dtype=self.dtype)

        key = (self.head_dim, self.rope_base, max_delta, self.device)
        _GLOBAL_COS_SIN_CACHE[key] = (cos_cache, sin_cache, max_delta)

    @classmethod
    def from_pretrained_sharded(
        cls,
        model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct",
        max_num_blocks: int = 1024,
        bridge_top_k: float = 0.08,
    ) -> "Qwen2DAGKVEngine":
        """Loads a large model (e.g. Qwen2.5-7B-Instruct) sharded across cuda:0 and cuda:1 in FP16."""
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            dtype=torch.float16,
            device_map="auto",
        )
        model.eval()
        return cls(
            model=model,
            tokenizer=tokenizer,
            max_num_blocks=max_num_blocks,
            device="cuda:0",
            bridge_top_k=bridge_top_k,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str = "cuda:0",
        max_num_blocks: int = 2048,
        bridge_top_k: float = 0.08,
    ) -> "Qwen2DAGKVEngine":
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            dtype=torch.float16,
            device_map=device,
        )
        model.eval()
        return cls(
            model=model,
            tokenizer=tokenizer,
            max_num_blocks=max_num_blocks,
            device=device,
            bridge_top_k=bridge_top_k,
        )

    @torch.inference_mode()
    def prefill_canonical_chunk(
        self,
        content: Union[str, List[int], torch.Tensor],
        metadata: Optional[Dict] = None,
    ) -> CanonicalChunk:
        if isinstance(content, str):
            token_ids = self.tokenizer.encode(content, add_special_tokens=False)
            token_ids = torch.tensor(token_ids, dtype=torch.int64, device=self.device)
        elif isinstance(content, list):
            token_ids = torch.tensor(content, dtype=torch.int64, device=self.device)
        else:
            token_ids = content.to(self.device, dtype=torch.int64)

        if token_ids.dim() > 1:
            token_ids = token_ids.squeeze()

        chunk_hash = self.canonicalizer.compute_chunk_hash(token_ids)
        if chunk_hash in self._chunk_cache:
            return self._chunk_cache[chunk_hash]

        input_ids, attention_mask, position_ids = self.canonicalizer.prepare_prefill_input(
            token_ids, device=self.device
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            logits_to_keep=1,
        )

        pkv = outputs.past_key_values
        total_len = input_ids.shape[-1]
        gqa_ratio = self.num_heads // self.num_kv_heads

        layer_k, layer_v, layer_lse = [], [], []
        causal_mask = torch.triu(
            torch.full((total_len, total_len), float("-inf"), device=self.device), diagonal=1
        ).unsqueeze(0)

        for l_idx in range(self.num_layers):
            if hasattr(pkv, "layers"):
                k_l = pkv.layers[l_idx].keys.squeeze(0)
                v_l = pkv.layers[l_idx].values.squeeze(0)
            elif hasattr(pkv, "key_cache"):
                k_l = pkv.key_cache[l_idx].squeeze(0)
                v_l = pkv.value_cache[l_idx].squeeze(0)
            else:
                k_l = pkv[l_idx][0].squeeze(0)
                v_l = pkv[l_idx][1].squeeze(0)

            layer_k.append(k_l)
            layer_v.append(v_l)

            if l_idx == 1:
                k_exp = k_l.repeat_interleave(gqa_ratio, dim=0) if gqa_ratio > 1 else k_l
                scores = torch.matmul(k_exp.to(torch.float32), k_exp.to(torch.float32).transpose(-1, -2)) * self.sm_scale
                lse_l = torch.logsumexp(scores + causal_mask, dim=-1)
            else:
                lse_l = torch.zeros((self.num_heads, total_len), dtype=torch.float32, device=self.device)
            layer_lse.append(lse_l)

        all_k = torch.stack(layer_k, dim=0)
        all_v = torch.stack(layer_v, dim=0)
        all_lse = torch.stack(layer_lse, dim=0)

        if self._sink_phys_block is None:
            s_len = self._num_sink
            sink_blk = self.block_manager.allocate_block(is_canonical=True, chunk_id="__canonical_sink__")
            self.block_manager.k_pool[sink_blk, :, :, :s_len, :] = all_k[:, :, :s_len, :].to(self.dtype)
            self.block_manager.v_pool[sink_blk, :, :, :s_len, :] = all_v[:, :, :s_len, :].to(self.dtype)
            self._sink_phys_block = sink_blk

        canonical_chunk = self.canonicalizer.canonicalize_kv(
            full_k=all_k,
            full_v=all_v,
            full_lse=all_lse,
            chunk_token_ids=token_ids,
            metadata=metadata,
        )

        self.block_manager.register_canonical_chunk(canonical_chunk)
        self._chunk_cache[chunk_hash] = canonical_chunk
        return canonical_chunk

    def assemble_prompt_dag(
        self,
        chunks: List[CanonicalChunk],
        seq_id: str = "seq_0",
    ) -> Tuple[OffsetBlockTable, List[int]]:
        chunk_sequence = []
        global_pos = self._num_sink if self._sink_phys_block is not None else 0
        all_tokens = list(self.canonicalizer.sink_token_ids) if self._sink_phys_block is not None else []

        for chunk in chunks:
            chunk_sequence.append((chunk, global_pos))
            global_pos += chunk.length
            all_tokens.extend(chunk.token_ids.tolist())

        table = self.block_manager.assemble_dag_sequence(seq_id=seq_id, chunk_sequence=chunk_sequence)
        if self._sink_phys_block is not None:
            from dag_kv.core.block_manager import LogicalBlock
            self.block_manager.physical_metadata[self._sink_phys_block].ref_count += 1
            sink_logical = LogicalBlock(
                logical_idx=0,
                physical_idx=self._sink_phys_block,
                delta=0,
                is_cow_shadow=False,
                valid_tokens=self._num_sink,
            )
            table.blocks.insert(0, sink_logical)
            for idx, blk in enumerate(table.blocks):
                blk.logical_idx = idx
        return table, all_tokens

    @torch.inference_mode()
    def repair_bridge_tokens(
        self,
        table: OffsetBlockTable,
        chunks: List[CanonicalChunk],
        top_k_percent: Optional[float] = None,
    ) -> Dict:
        """Executes Layer-1 LSE Probe and Fused Single-Pass Triton Micro-Prefill on Bridge Tokens."""
        k_pct = top_k_percent if top_k_percent is not None else self.bridge_top_k
        if len(chunks) <= 1 or k_pct <= 0.0:
            return {"num_repaired": 0, "bridge_results": []}

        global_offset = (self._num_sink if self._sink_phys_block is not None else 0) + chunks[0].length
        repaired_total = 0
        all_bridge_results = []

        all_token_ids = []
        all_global_pos = []
        all_canon_pos = []

        gqa = self.num_heads // self.num_kv_heads

        # 1. Select bridge tokens across all chunks using Layer-1 LSE Probe
        for chunk_idx in range(1, len(chunks)):
            target_chunk = chunks[chunk_idx]
            l_target = target_chunk.length

            k_prior_list = [prior_c.k_cache[1] for prior_c in chunks[:chunk_idx]]
            k_prior = torch.cat(k_prior_list, dim=1).to(self.device)

            q_target = target_chunk.k_cache[1].to(self.device)
            q_probe = q_target.repeat_interleave(gqa, dim=0) if q_target.shape[0] != self.num_heads else q_target
            lse_local = target_chunk.lse_local[1].to(self.device)

            alpha = compute_lse_perturbation(q_probe, k_prior, lse_local, scale=self.sm_scale)
            selection = select_bridge_tokens(
                alpha=alpha,
                top_k_percent=k_pct,
                chunk_global_offset=global_offset,
            )

            all_bridge_results.append(selection)
            repaired_total += selection.num_bridge_tokens

            if selection.num_bridge_tokens > 0:
                b_local = selection.bridge_indices.to(self.device)
                b_global = selection.global_bridge_indices.to(self.device)
                all_token_ids.append(target_chunk.token_ids.to(self.device)[b_local])
                all_global_pos.append(b_global)
                all_canon_pos.append(4 + b_local)

            global_offset += l_target

        if repaired_total == 0:
            return {"num_repaired": 0, "bridge_results": all_bridge_results}

        # 2. Vectorized CoW Shadow Block allocation & slot mapping (Done ONCE before Layer 0)
        cat_token_ids = torch.cat(all_token_ids, dim=0)
        cat_global_pos = torch.cat(all_global_pos, dim=0)
        cat_canon_pos = torch.cat(all_canon_pos, dim=0)

        phys_ids_list = []
        slot_offs_list = []
        for g_i in cat_global_pos.tolist():
            cum = 0
            for l_b_idx, b in enumerate(table.blocks):
                if cum <= g_i < cum + b.valid_tokens:
                    p_id = self.block_manager.create_cow_shadow_block(table, l_b_idx)
                    phys_ids_list.append(p_id)
                    slot_offs_list.append(g_i - cum)
                    break
                cum += b.valid_tokens

        bridge_phys_ids = torch.tensor(phys_ids_list, dtype=torch.long, device=self.device)
        bridge_slot_offs = torch.tensor(slot_offs_list, dtype=torch.long, device=self.device)

        # Pre-build GPU PagedAttention tables once before the 36-layer loop
        block_tables_2d = table.get_physical_indices(device=self.device).unsqueeze(0)
        offset_tables_2d = table.get_deltas(device=self.device).unsqueeze(0)
        block_starts_2d = table.get_block_starts(device=self.device).unsqueeze(0)
        valid_lens_2d = table.get_valid_tokens(device=self.device).unsqueeze(0)
        bridge_pos_2d = cat_global_pos.to(torch.int32).unsqueeze(0)

        # Precompute RoPE cos/sin for global Q and canonical K once before the 36-layer loop
        ang_q = cat_global_pos.to(torch.float32).unsqueeze(-1) * self.inv_freq.unsqueeze(0)
        ang_q = torch.cat([ang_q, ang_q], dim=-1)
        cos_q = torch.cos(ang_q).to(self.dtype).unsqueeze(0)  # (1, N_bridge, head_dim)
        sin_q = torch.sin(ang_q).to(self.dtype).unsqueeze(0)

        ang_k = cat_canon_pos.to(torch.float32).unsqueeze(-1) * self.inv_freq.unsqueeze(0)
        ang_k = torch.cat([ang_k, ang_k], dim=-1)
        cos_k = torch.cos(ang_k).to(self.dtype).unsqueeze(0)  # (1, N_bridge, head_dim)
        sin_k = torch.sin(ang_k).to(self.dtype).unsqueeze(0)

        # 3. Single Fused 36-Layer Pass over all N_bridge tokens via Triton Sparse Stitch Kernel
        n_b = cat_token_ids.shape[0]
        bridge_h = self.model.get_input_embeddings()(cat_token_ids).to(self.dtype)

        for l_idx, layer in enumerate(self.model.model.layers):
            res = bridge_h
            h_norm = layer.input_layernorm(bridge_h)
            attn = layer.self_attn

            q = attn.q_proj(h_norm).view(n_b, self.num_heads, self.head_dim).transpose(0, 1)
            k = attn.k_proj(h_norm).view(n_b, self.num_kv_heads, self.head_dim).transpose(0, 1)
            v = attn.v_proj(h_norm).view(n_b, self.num_kv_heads, self.head_dim).transpose(0, 1)

            q_rot = (q * cos_q) + (rotate_half(q) * sin_q)
            k_rot = (k * cos_k) + (rotate_half(k) * sin_k)

            # Vectorized scatter into CoW Shadow Blocks
            if self.block_manager.quantization == "int8":
                from dag_kv.core.quantizer import KVQuantizer
                k_q, k_sc = KVQuantizer.quantize_int8(k_rot.transpose(0, 1), dim=-1)
                v_q, v_sc = KVQuantizer.quantize_int8(v.transpose(0, 1), dim=-1)
                self.block_manager.k_pool[bridge_phys_ids, l_idx, :, bridge_slot_offs, :] = k_q
                self.block_manager.v_pool[bridge_phys_ids, l_idx, :, bridge_slot_offs, :] = v_q
                self.block_manager.k_scale_pool[bridge_phys_ids, l_idx, :, bridge_slot_offs, :] = k_sc
                self.block_manager.v_scale_pool[bridge_phys_ids, l_idx, :, bridge_slot_offs, :] = v_sc
                k_scales_l = self.block_manager.k_scale_pool[:, l_idx]
                v_scales_l = self.block_manager.v_scale_pool[:, l_idx]
            else:
                self.block_manager.k_pool[bridge_phys_ids, l_idx, :, bridge_slot_offs, :] = k_rot.transpose(0, 1)
                self.block_manager.v_pool[bridge_phys_ids, l_idx, :, bridge_slot_offs, :] = v.transpose(0, 1)
                k_scales_l = None
                v_scales_l = None

            # Universal Sparse Stitch Kernel (Triton on CUDA, PyTorch on CPU/MPS/INT8)
            out_b = sparse_bridge_attention(
                q_bridge=q_rot.unsqueeze(0),
                bridge_positions=bridge_pos_2d,
                k_pool=self.block_manager.k_pool[:, l_idx],
                v_pool=self.block_manager.v_pool[:, l_idx],
                block_tables=block_tables_2d,
                offset_tables=offset_tables_2d,
                sm_scale=self.sm_scale,
                base=self.rope_base,
                block_start_tables=block_starts_2d,
                valid_lens_tables=valid_lens_2d,
                k_scales=k_scales_l,
                v_scales=v_scales_l,
            )

            out_b = out_b.squeeze(0).transpose(0, 1).reshape(n_b, -1)
            bridge_h = res + attn.o_proj(out_b)
            bridge_h = bridge_h + layer.mlp(layer.post_attention_layernorm(bridge_h))

        last_hidden = self.model.model.norm(bridge_h[-1:, :])
        last_logits = self.model.lm_head(last_hidden).squeeze(0)

        return {
            "num_repaired": repaired_total,
            "bridge_results": all_bridge_results,
            "last_logits": last_logits,
        }

    @torch.inference_mode()
    def append_and_decode(
        self,
        table: OffsetBlockTable,
        token_id: int,
        current_pos: int,
    ) -> torch.Tensor:
        """Appends new token K/V to table and evaluates single-token decode logits via Triton."""
        if len(table.blocks) == 0:
            new_blk = self.block_manager.allocate_block()
            table.append_block(physical_idx=new_blk, delta=0, valid_tokens=0)

        active_blk = table.blocks[-1]
        is_canonical_blk = self.block_manager.physical_metadata[active_blk.physical_idx].is_canonical
        if active_blk.valid_tokens >= self.block_size or active_blk.delta != 0 or is_canonical_blk:
            new_blk = self.block_manager.allocate_block(is_canonical=False)
            table.append_block(physical_idx=new_blk, delta=0, valid_tokens=0)
            active_blk = table.blocks[-1]

        active_blk.valid_tokens += 1
        blk_offset = active_blk.valid_tokens - 1
        phys_blk = active_blk.physical_idx

        input_ids = torch.tensor([[token_id]], dtype=torch.int64, device=self.device)
        hidden_states = self.model.get_input_embeddings()(input_ids).to(dtype=self.dtype)

        pos_tensor = torch.tensor([[current_pos]], dtype=torch.long, device=self.device)
        cos_n, sin_n = self.model.model.rotary_emb(hidden_states, pos_tensor)

        # Build PagedAttention metadata tensors once before the 36-layer loop
        block_tables_2d = table.get_physical_indices(device=self.device).unsqueeze(0)
        offset_tables_2d = table.get_deltas(device=self.device).unsqueeze(0)
        block_starts_2d = table.get_block_starts(device=self.device).unsqueeze(0)
        valid_lens_2d = table.get_valid_tokens(device=self.device).unsqueeze(0)
        seq_lens_1d = torch.tensor([current_pos + 1], dtype=torch.int32, device=self.device)

        for layer_idx, layer in enumerate(self.model.model.layers):
            res = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)

            attn = layer.self_attn
            q = attn.q_proj(hidden_states).view(1, 1, self.num_heads, self.head_dim).transpose(1, 2)
            k = attn.k_proj(hidden_states).view(1, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = attn.v_proj(hidden_states).view(1, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)

            q_rot, k_rot = apply_rotary_pos_emb(q, k, cos_n, sin_n)

            if self.block_manager.quantization == "int8":
                from dag_kv.core.quantizer import KVQuantizer
                k_q, k_sc = KVQuantizer.quantize_int8(k_rot.squeeze(0).squeeze(1), dim=-1)
                v_q, v_sc = KVQuantizer.quantize_int8(v.squeeze(0).squeeze(1), dim=-1)
                self.block_manager.k_pool[phys_blk, layer_idx, :, blk_offset, :] = k_q
                self.block_manager.v_pool[phys_blk, layer_idx, :, blk_offset, :] = v_q
                self.block_manager.k_scale_pool[phys_blk, layer_idx, :, blk_offset, :] = k_sc
                self.block_manager.v_scale_pool[phys_blk, layer_idx, :, blk_offset, :] = v_sc
                k_scales_l = self.block_manager.k_scale_pool[:, layer_idx]
                v_scales_l = self.block_manager.v_scale_pool[:, layer_idx]
            else:
                self.block_manager.k_pool[phys_blk, layer_idx, :, blk_offset, :] = k_rot.squeeze(0).squeeze(1)
                self.block_manager.v_pool[phys_blk, layer_idx, :, blk_offset, :] = v.squeeze(0).squeeze(1)
                k_scales_l = None
                v_scales_l = None

            attn_out = adjoint_paged_attention(
                q=q_rot,
                k_pool=self.block_manager.k_pool[:, layer_idx],
                v_pool=self.block_manager.v_pool[:, layer_idx],
                block_tables=block_tables_2d,
                offset_tables=offset_tables_2d,
                seq_lens=seq_lens_1d,
                sm_scale=self.sm_scale,
                base=self.rope_base,
                block_start_tables=block_starts_2d,
                valid_lens_tables=valid_lens_2d,
                k_scales=k_scales_l,
                v_scales=v_scales_l,
            )

            attn_out = attn_out.transpose(1, 2).reshape(1, 1, -1)
            hidden_states = res + attn.o_proj(attn_out)
            hidden_states = hidden_states + layer.mlp(layer.post_attention_layernorm(hidden_states))

        hidden_states = self.model.model.norm(hidden_states)
        logits = self.model.lm_head(hidden_states).squeeze(0).squeeze(0)
        return logits

    @torch.inference_mode()
    def generate(
        self,
        chunks: List[CanonicalChunk],
        prompt_suffix: str = "",
        max_new_tokens: int = 16,
        temperature: float = 0.0,
    ) -> Tuple[List[int], str]:
        table, doc_tokens = self.assemble_prompt_dag(chunks, seq_id="gen_seq")
        self.repair_bridge_tokens(table, chunks)

        suffix_ids = self.tokenizer.encode(prompt_suffix, add_special_tokens=False)
        curr_pos = len(doc_tokens)

        last_logits = None
        for tok in suffix_ids:
            last_logits = self.append_and_decode(table, token_id=tok, current_pos=curr_pos)
            curr_pos += 1

        if last_logits is None:
            last_logits = self.append_and_decode(table, token_id=doc_tokens[-1], current_pos=curr_pos)
            curr_pos += 1

        generated = []
        for _ in range(max_new_tokens):
            if temperature == 0.0:
                next_tok = torch.argmax(last_logits, dim=-1).item()
            else:
                probs = torch.softmax(last_logits / temperature, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1).item()

            generated.append(next_tok)
            if next_tok == self.tokenizer.eos_token_id:
                break

            last_logits = self.append_and_decode(table, token_id=next_tok, current_pos=curr_pos)
            curr_pos += 1

        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return generated, text

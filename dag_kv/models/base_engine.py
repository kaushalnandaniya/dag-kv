"""Model-Agnostic Base Engine for Position-Decoupled DAG-KV Caching.

Abstracts attention layers, RoPE geometries, normalization layers, and MLPs
to support Qwen2, Llama-3, Mistral, and Gemma-2 across heterogeneous hardware.
"""

from abc import ABC, abstractmethod
import math
from typing import Dict, List, Optional, Tuple, Union
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from dag_kv.core.block_manager import BlockManager, OffsetBlockTable
from dag_kv.core.lse_probe import LSEProbe, compute_lse_perturbation, select_bridge_tokens
from dag_kv.core.rope_adjoint import compute_inv_freq, rotate_half
from dag_kv.core.sink_canonicalizer import CanonicalChunk, SinkCanonicalizer
from dag_kv.devices.device_manager import get_device_manager
from dag_kv.devices.shard_config import ShardConfig, QuantizationMode
from dag_kv.kernels import (
    adjoint_paged_attention,
    sparse_bridge_attention,
    _GLOBAL_COS_SIN_CACHE,
)


class DAGKVEngine(ABC):
    """Abstract Base Class for DAG-KV inference engines across LLM architectures."""

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
        enable_lru: bool = True,
        disk_store: Optional[object] = None,
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
        self.num_layers = self._get_num_layers()
        self.num_heads = self._get_num_heads()
        self.num_kv_heads = self._get_num_kv_heads()
        self.head_dim = self._get_head_dim()
        self.sm_scale = 1.0 / math.sqrt(self.head_dim)

        self.rope_base = self._get_rope_base()
        self.inv_freq = self._get_inv_freq()
        self._init_rope_cache()

        sink_prefix, sink_ids = self._get_sink_config()
        self.canonicalizer = SinkCanonicalizer(
            sink_prefix=sink_prefix,
            num_sink_tokens=4,
            sink_token_ids=sink_ids,
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
            enable_lru=enable_lru,
            disk_store=disk_store,
        )

        self.lse_probe = LSEProbe(top_k_percent=self.bridge_top_k, probe_layer=1)
        self._chunk_cache: Dict[str, CanonicalChunk] = {}
        self._sink_phys_block: Optional[int] = None
        self._num_sink: int = self.canonicalizer.num_sink_tokens

    # --- Architecture Hooks (Customizable by Subclasses) ---

    def _get_num_layers(self) -> int:
        return self.config.num_hidden_layers

    def _get_num_heads(self) -> int:
        return self.config.num_attention_heads

    def _get_num_kv_heads(self) -> int:
        return getattr(self.config, "num_key_value_heads", self.config.num_attention_heads)

    def _get_head_dim(self) -> int:
        return getattr(self.config, "head_dim", self.config.hidden_size // self.num_heads)

    def _get_rope_base(self) -> float:
        return getattr(self.config, "rope_theta", 10000.0) or 10000.0

    def _get_inv_freq(self) -> torch.Tensor:
        rot_emb = getattr(getattr(self.model, "model", self.model), "rotary_emb", None)
        if rot_emb is not None and hasattr(rot_emb, "inv_freq"):
            return rot_emb.inv_freq.to(self.device)
        return compute_inv_freq(self.head_dim, base=self.rope_base, device=self.device)

    def _get_sink_config(self) -> Tuple[str, Optional[List[int]]]:
        vocab_size = getattr(self.config, "vocab_size", 100000)
        sink_ids = [min(i, vocab_size - 1) for i in range(4)]
        return "<|im_start|>system\n", sink_ids

    def get_layers(self):
        return self.model.model.layers

    def get_input_layernorm(self, layer, h: torch.Tensor) -> torch.Tensor:
        return layer.input_layernorm(h)

    def compute_qkv(self, layer, h_norm: torch.Tensor, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn = layer.self_attn
        q = attn.q_proj(h_norm).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = attn.k_proj(h_norm).view(batch_size, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = attn.v_proj(h_norm).view(batch_size, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        return q, k, v

    def apply_attn_output_and_mlp(self, layer, res: torch.Tensor, attn_out: torch.Tensor) -> torch.Tensor:
        h = res + layer.self_attn.o_proj(attn_out)
        h = h + layer.mlp(layer.post_attention_layernorm(h))
        return h

    def get_final_norm(self, h: torch.Tensor) -> torch.Tensor:
        return self.model.model.norm(h)

    def get_lm_head(self, h: torch.Tensor) -> torch.Tensor:
        return self.model.lm_head(h)

    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings()(token_ids).to(self.dtype)

    def _init_rope_cache(self):
        max_delta = 16384
        deltas = torch.arange(-max_delta, max_delta + 1, device=self.device, dtype=torch.float32)
        angles = deltas.unsqueeze(-1) * self.inv_freq.unsqueeze(0)
        angles = torch.cat([angles, angles], dim=-1)
        cos_cache = torch.cos(angles).to(dtype=self.dtype)
        sin_cache = torch.sin(angles).to(dtype=self.dtype)
        key = (self.head_dim, self.rope_base, max_delta, self.device)
        _GLOBAL_COS_SIN_CACHE[key] = (cos_cache, sin_cache, max_delta)

    # --- Canonical Prefill, DAG Stitching, and Bridge Repair ---

    @torch.inference_mode()
    def prefill_canonical_chunk(
        self,
        content: Union[str, List[int], torch.Tensor],
        metadata: Optional[Dict] = None,
    ) -> CanonicalChunk:
        """Prefills and caches a document chunk starting at canonical position 4."""
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

        # Check disk store if available
        if self.block_manager.disk_store is not None and hasattr(self.block_manager.disk_store, "load_chunk"):
            disk_chunk = self.block_manager.disk_store.load_chunk(chunk_hash, device=self.device, dtype=self.dtype)
            if disk_chunk is not None:
                self.block_manager.register_canonical_chunk(disk_chunk)
                self._chunk_cache[chunk_hash] = disk_chunk
                return disk_chunk

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
            sink_k = all_k[:, :, :s_len, :].to(self.device, dtype=self.dtype)
            sink_v = all_v[:, :, :s_len, :].to(self.device, dtype=self.dtype)
            if self.block_manager.quantization == "int8":
                from dag_kv.core.quantizer import KVQuantizer
                qk, sk = KVQuantizer.quantize_int8(sink_k, dim=-1)
                qv, sv = KVQuantizer.quantize_int8(sink_v, dim=-1)
                self.block_manager.k_pool[sink_blk, :, :, :s_len, :] = qk
                self.block_manager.v_pool[sink_blk, :, :, :s_len, :] = qv
                self.block_manager.k_scale_pool[sink_blk, :, :, :s_len, :] = sk
                self.block_manager.v_scale_pool[sink_blk, :, :, :s_len, :] = sv
            elif self.block_manager.quantization == "int4":
                from dag_kv.core.quantizer import KVQuantizer
                qk, sk = KVQuantizer.quantize_int4(sink_k, dim=-1)
                qv, sv = KVQuantizer.quantize_int4(sink_v, dim=-1)
                self.block_manager.k_pool[sink_blk, :, :, :s_len, :] = qk
                self.block_manager.v_pool[sink_blk, :, :, :s_len, :] = qv
                self.block_manager.k_scale_pool[sink_blk, :, :, :s_len, :] = sk
                self.block_manager.v_scale_pool[sink_blk, :, :, :s_len, :] = sv
            else:
                self.block_manager.k_pool[sink_blk, :, :, :s_len, :] = sink_k
                self.block_manager.v_pool[sink_blk, :, :, :s_len, :] = sink_v
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
        """Assembles logical block table for a sequence of chunks."""
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
        """Executes Layer-1 LSE Probe and sparse micro-prefill repair on boundary tokens."""
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

        for chunk_idx in range(1, len(chunks)):
            target_chunk = chunks[chunk_idx]
            l_target = target_chunk.length

            k_prior_list = [prior_c.k_cache[1].to(self.device) for prior_c in chunks[:chunk_idx]]
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

        block_tables_2d = table.get_physical_indices(device=self.device).unsqueeze(0)
        offset_tables_2d = table.get_deltas(device=self.device).unsqueeze(0)
        block_starts_2d = table.get_block_starts(device=self.device).unsqueeze(0)
        valid_lens_2d = table.get_valid_tokens(device=self.device).unsqueeze(0)
        bridge_pos_2d = cat_global_pos.to(torch.int32).unsqueeze(0)

        ang_q = cat_global_pos.to(torch.float32).unsqueeze(-1) * self.inv_freq.unsqueeze(0)
        ang_q = torch.cat([ang_q, ang_q], dim=-1)
        cos_q = torch.cos(ang_q).to(self.dtype).unsqueeze(0)
        sin_q = torch.sin(ang_q).to(self.dtype).unsqueeze(0)

        ang_k = cat_canon_pos.to(torch.float32).unsqueeze(-1) * self.inv_freq.unsqueeze(0)
        ang_k = torch.cat([ang_k, ang_k], dim=-1)
        cos_k = torch.cos(ang_k).to(self.dtype).unsqueeze(0)
        sin_k = torch.sin(ang_k).to(self.dtype).unsqueeze(0)

        n_b = cat_token_ids.shape[0]
        bridge_h = self.embed_tokens(cat_token_ids)

        for l_idx, layer in enumerate(self.get_layers()):
            res = bridge_h
            h_norm = self.get_input_layernorm(layer, bridge_h)
            attn = layer.self_attn

            q = attn.q_proj(h_norm).view(n_b, self.num_heads, self.head_dim).transpose(0, 1)
            k = attn.k_proj(h_norm).view(n_b, self.num_kv_heads, self.head_dim).transpose(0, 1)
            v = attn.v_proj(h_norm).view(n_b, self.num_kv_heads, self.head_dim).transpose(0, 1)

            q_rot = (q * cos_q) + (rotate_half(q) * sin_q)
            k_rot = (k * cos_k) + (rotate_half(k) * sin_k)

            if self.block_manager.quantization == "int8":
                from dag_kv.core.quantizer import KVQuantizer
                kq, ksc = KVQuantizer.quantize_int8(k_rot.transpose(0, 1), dim=-1)
                vq, vsc = KVQuantizer.quantize_int8(v.transpose(0, 1), dim=-1)
                self.block_manager.k_pool[bridge_phys_ids, l_idx, :, bridge_slot_offs, :] = kq
                self.block_manager.v_pool[bridge_phys_ids, l_idx, :, bridge_slot_offs, :] = vq
                self.block_manager.k_scale_pool[bridge_phys_ids, l_idx, :, bridge_slot_offs, :] = ksc
                self.block_manager.v_scale_pool[bridge_phys_ids, l_idx, :, bridge_slot_offs, :] = vsc
                k_scales_l = self.block_manager.k_scale_pool[:, l_idx]
                v_scales_l = self.block_manager.v_scale_pool[:, l_idx]
            elif self.block_manager.quantization == "int4":
                from dag_kv.core.quantizer import KVQuantizer
                kq, ksc = KVQuantizer.quantize_int4(k_rot.transpose(0, 1), dim=-1)
                vq, vsc = KVQuantizer.quantize_int4(v.transpose(0, 1), dim=-1)
                self.block_manager.k_pool[bridge_phys_ids, l_idx, :, bridge_slot_offs, :] = kq
                self.block_manager.v_pool[bridge_phys_ids, l_idx, :, bridge_slot_offs, :] = vq
                self.block_manager.k_scale_pool[bridge_phys_ids, l_idx, :, bridge_slot_offs, :] = ksc
                self.block_manager.v_scale_pool[bridge_phys_ids, l_idx, :, bridge_slot_offs, :] = vsc
                k_scales_l = self.block_manager.k_scale_pool[:, l_idx]
                v_scales_l = self.block_manager.v_scale_pool[:, l_idx]
            else:
                self.block_manager.k_pool[bridge_phys_ids, l_idx, :, bridge_slot_offs, :] = k_rot.transpose(0, 1)
                self.block_manager.v_pool[bridge_phys_ids, l_idx, :, bridge_slot_offs, :] = v.transpose(0, 1)
                k_scales_l = None
                v_scales_l = None

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
            bridge_h = self.apply_attn_output_and_mlp(layer, res, out_b)

        last_hidden = self.get_final_norm(bridge_h[-1:, :])
        last_logits = self.get_lm_head(last_hidden).squeeze(0)

        return {
            "num_repaired": repaired_total,
            "bridge_results": all_bridge_results,
            "last_logits": last_logits,
        }

    # --- Batched & Single Decode Execution ---

    @torch.inference_mode()
    def batched_append_and_decode(
        self,
        tables: List[OffsetBlockTable],
        token_ids: List[int],
        current_positions: List[int],
    ) -> torch.Tensor:
        """Executes a single forward step across B concurrent sequences."""
        B = len(tables)
        phys_list, slot_list = [], []

        for i, table in enumerate(tables):
            cur_pos = current_positions[i]
            if len(table.blocks) == 0 or table.blocks[-1].valid_tokens == self.block_size or not table.blocks[-1].is_cow_shadow:
                p = self.block_manager.allocate_block(is_canonical=False)
                table.append_block(physical_idx=p, delta=0, valid_tokens=1, is_cow_shadow=True)
                slot_in_blk = 0
            else:
                last_blk = table.blocks[-1]
                p = last_blk.physical_idx
                slot_in_blk = last_blk.valid_tokens
                last_blk.valid_tokens += 1
            phys_list.append(p)
            slot_list.append(slot_in_blk)

        max_blks = max(len(t.blocks) for t in tables)
        bt = torch.zeros((B, max_blks), dtype=torch.int32, device=self.device)
        ot = torch.zeros((B, max_blks), dtype=torch.int32, device=self.device)
        bs = torch.zeros((B, max_blks), dtype=torch.int32, device=self.device)
        vl = torch.zeros((B, max_blks), dtype=torch.int32, device=self.device)
        sl = torch.zeros((B,), dtype=torch.int32, device=self.device)

        for i, t in enumerate(tables):
            n_b = len(t.blocks)
            bt[i, :n_b] = t.get_physical_indices(device=self.device)
            ot[i, :n_b] = t.get_deltas(device=self.device)
            bs[i, :n_b] = t.get_block_starts(device=self.device)
            vl[i, :n_b] = t.get_valid_tokens(device=self.device)
            sl[i] = current_positions[i] + 1

        pos_t = torch.tensor(current_positions, dtype=torch.float32, device=self.device)
        ang = pos_t.unsqueeze(-1) * self.inv_freq.unsqueeze(0)
        ang = torch.cat([ang, ang], dim=-1)
        cos = torch.cos(ang).to(self.dtype).view(B, 1, 1, self.head_dim)
        sin = torch.sin(ang).to(self.dtype).view(B, 1, 1, self.head_dim)

        phys_t = torch.tensor(phys_list, dtype=torch.long, device=self.device)
        slots_t = torch.tensor(slot_list, dtype=torch.long, device=self.device)

        inp = torch.tensor(token_ids, dtype=torch.long, device=self.device).unsqueeze(1)
        h = self.embed_tokens(inp)

        for l_idx, layer in enumerate(self.get_layers()):
            res = h
            h_norm = self.get_input_layernorm(layer, h)
            q, k, v = self.compute_qkv(layer, h_norm, B)

            q_rot = (q * cos) + (rotate_half(q) * sin)
            k_rot = (k * cos) + (rotate_half(k) * sin)

            if self.block_manager.quantization == "int8":
                from dag_kv.core.quantizer import KVQuantizer
                kq, ksc = KVQuantizer.quantize_int8(k_rot.squeeze(2), dim=-1)
                vq, vsc = KVQuantizer.quantize_int8(v.squeeze(2), dim=-1)
                self.block_manager.k_pool[phys_t, l_idx, :, slots_t, :] = kq
                self.block_manager.v_pool[phys_t, l_idx, :, slots_t, :] = vq
                self.block_manager.k_scale_pool[phys_t, l_idx, :, slots_t, :] = ksc
                self.block_manager.v_scale_pool[phys_t, l_idx, :, slots_t, :] = vsc
                k_scales_l = self.block_manager.k_scale_pool[:, l_idx]
                v_scales_l = self.block_manager.v_scale_pool[:, l_idx]
            elif self.block_manager.quantization == "int4":
                from dag_kv.core.quantizer import KVQuantizer
                kq, ksc = KVQuantizer.quantize_int4(k_rot.squeeze(2), dim=-1)
                vq, vsc = KVQuantizer.quantize_int4(v.squeeze(2), dim=-1)
                self.block_manager.k_pool[phys_t, l_idx, :, slots_t, :] = kq
                self.block_manager.v_pool[phys_t, l_idx, :, slots_t, :] = vq
                self.block_manager.k_scale_pool[phys_t, l_idx, :, slots_t, :] = ksc
                self.block_manager.v_scale_pool[phys_t, l_idx, :, slots_t, :] = vsc
                k_scales_l = self.block_manager.k_scale_pool[:, l_idx]
                v_scales_l = self.block_manager.v_scale_pool[:, l_idx]
            else:
                self.block_manager.k_pool[phys_t, l_idx, :, slots_t, :] = k_rot.squeeze(2)
                self.block_manager.v_pool[phys_t, l_idx, :, slots_t, :] = v.squeeze(2)
                k_scales_l = None
                v_scales_l = None

            attn_out = adjoint_paged_attention(
                q=q_rot,
                k_pool=self.block_manager.k_pool[:, l_idx],
                v_pool=self.block_manager.v_pool[:, l_idx],
                block_tables=bt,
                offset_tables=ot,
                seq_lens=sl,
                sm_scale=self.sm_scale,
                base=self.rope_base,
                block_start_tables=bs,
                valid_lens_tables=vl,
                k_scales=k_scales_l,
                v_scales=v_scales_l,
            )
            attn_out = attn_out.transpose(1, 2).reshape(B, 1, -1)
            h = self.apply_attn_output_and_mlp(layer, res, attn_out)

        final_norm = self.get_final_norm(h)
        logits = self.get_lm_head(final_norm).squeeze(1)  # (B, vocab_size)
        return logits

    @torch.inference_mode()
    def append_and_decode(
        self,
        table: OffsetBlockTable,
        token_id: int,
        current_pos: int,
    ) -> torch.Tensor:
        """Appends new token K/V to table and evaluates single-token decode logits."""
        logits = self.batched_append_and_decode([table], [token_id], [current_pos])
        return logits.squeeze(0)  # (vocab_size,)

    @torch.inference_mode()
    def generate(
        self,
        chunks: List[CanonicalChunk],
        prompt_suffix: str = "",
        max_new_tokens: int = 16,
        temperature: float = 0.0,
    ) -> str:
        """Generates completion text conditioned on stitched canonical document chunks."""
        import uuid
        seq_id = f"gen_{uuid.uuid4().hex[:8]}"
        table, doc_tokens = self.assemble_prompt_dag(chunks, seq_id=seq_id)
        try:
            self.repair_bridge_tokens(table, chunks)

            suffix_ids = self.tokenizer.encode(prompt_suffix, add_special_tokens=False) if prompt_suffix else []
            cur_pos = len(doc_tokens)
            logits = None

            for s_tok in suffix_ids:
                logits = self.append_and_decode(table, token_id=s_tok, current_pos=cur_pos)
                cur_pos += 1

            if logits is None and len(suffix_ids) == 0:
                last_tok = doc_tokens[-1] if doc_tokens else 0
                logits = self.append_and_decode(table, token_id=last_tok, current_pos=cur_pos)
                cur_pos += 1

            generated_ids = []
            for _ in range(max_new_tokens):
                if temperature <= 0.0:
                    next_tok = int(torch.argmax(logits, dim=-1).item())
                else:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_tok = int(torch.multinomial(probs, num_samples=1).item())

                generated_ids.append(next_tok)
                if next_tok == getattr(self.tokenizer, "eos_token_id", None):
                    break

                logits = self.append_and_decode(table, token_id=next_tok, current_pos=cur_pos)
                cur_pos += 1

            return generated_ids, self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        finally:
            self.block_manager.free_sequence(table)

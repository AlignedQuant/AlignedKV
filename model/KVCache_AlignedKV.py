import math
from typing import Optional, Dict, Any, Tuple, List

import torch
import torch.nn.functional as F
from transformers import PretrainedConfig
from transformers.cache_utils import Cache
from transformers.utils import is_torchdynamo_compiling, logging

from extension.k_cache_file import K_Cache_Class
from extension.v_cache_file import V_Cache_Class

logger = logging.get_logger(__name__)


# 注意是所有层共享一个cache
# 在模型generate时直接传入就好
class QuantizedCache_AlignedKV(Cache):
    def __init__(self, config: PretrainedConfig, max_batch_size: int, max_cache_len: int, device, dtype=None) -> None:
        super().__init__()
        self.max_batch_size = max_batch_size
        self.max_cache_len = config.max_position_embeddings if max_cache_len is None else max_cache_len
        # Some model define a custom `head_dim` != config.hidden_size // config.num_attention_heads
        self.head_dim = (
            config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads
        )

        self.dtype = dtype if dtype is not None else torch.float32
        self.num_key_value_heads = (
            config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        )
        assert config.num_attention_heads == self.num_key_value_heads, "We don't support GQA yet."

        self.key_cache: List[K_Cache_Class] = []
        self.value_cache: List[V_Cache_Class] = []
        # Note: There will be significant perf decrease if switching to use 5D tensors instead.
        cache_shape = (max_batch_size, self.num_key_value_heads, self.max_cache_len, self.head_dim)
        for idx in range(config.num_hidden_layers):
            if_register_buffer = is_torchdynamo_compiling()
            new_key_cache = K_Cache_Class(max_batch_size, self.max_cache_len, self.num_key_value_heads, self.head_dim,
                                          idx, device, if_register_buffer)
            new_value_cache = V_Cache_Class(max_batch_size, self.max_cache_len, self.num_key_value_heads, self.head_dim,
                                            idx, device, if_register_buffer)
            self.key_cache.append(new_key_cache)
            self.value_cache.append(new_value_cache)
        self.reference = False  # debug using

    def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise Exception("This function is not used in AlignedKV")

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states that were seen by the model."""
        return self.key_cache[layer_idx].current_cache_len

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states."""
        return self.max_cache_len

    def reset(self):
        """Resets the cache values while preserving the objects"""
        for layer_idx in range(len(self.key_cache)):
            # In-place ops prevent breaking the static address
            self.key_cache[layer_idx].zerolize()
            self.value_cache[layer_idx].zerolize()

    def prefill_k(self, q: torch.Tensor, k: torch.Tensor, seqlen: int, layer_id: int) -> torch.Tensor:
        # input: q.shape = (bsz, seqlen, n_local_kv_heads, head_dim)
        # input: k.shape = (bsz, seqlen, n_local_kv_heads, head_dim)
        # output: score.shape = (bsz, n_local_kv_heads, seqlen, seqlen)
        self.key_cache[layer_id].save(k, 0, seqlen)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        return scores

    def prefill_v(self, scores: torch.Tensor, v: torch.Tensor, seqlen: int, layer_id: int) -> torch.Tensor:
        # input: scores.shape = (bsz, n_local_kv_heads, seqlen, seqlen)
        # input: v.shape = (bsz, seqlen, n_local_kv_heads, head_dim)
        # output: output.shape = (bsz, seqlen, n_local_kv_heads, head_dim)
        self.value_cache[layer_id].save(v, 0, seqlen)
        v = v.transpose(1, 2)
        output = torch.matmul(scores, v)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous()
        return output

    def prefill(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seqlen: int,
                mask: torch.Tensor, layer_id: int) -> torch.Tensor:
        scores = self.prefill_k(q, k, seqlen, layer_id)
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(q)
        output = self.prefill_v(scores, v, seqlen, layer_id)
        return output

    def decoding_k(self, q: torch.Tensor, k: torch.Tensor, start_pos: int, seqlen: int, layer_id: int) -> torch.Tensor:
        # input: q.shape = (bsz, seqlen, n_local_kv_heads, head_dim)
        # input: k.shape = (bsz, seqlen, n_local_kv_heads, head_dim)
        # output: score.shape = (bsz, n_local_kv_heads, seqlen, seqlen)
        self.key_cache[layer_id].save(k, start_pos, seqlen)
        return self.key_cache[layer_id].decoding_compute(q, start_pos, seqlen, self.reference)

    def decoding_v(self, scores: torch.Tensor, v: torch.Tensor, start_pos: int, seqlen: int,
                   layer_id: int) -> torch.Tensor:
        # input: scores.shape = (bsz, n_local_kv_heads, seqlen, seqlen)
        # input: v.shape = (bsz, seqlen, n_local_kv_heads, head_dim)
        # output: output.shape = (bsz, seqlen, n_local_kv_heads, head_dim)
        self.value_cache[layer_id].save(v, start_pos, seqlen)
        return self.value_cache[layer_id].decoding_compute(scores, start_pos, seqlen, self.reference)

    def decoding(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, start_pos: int, seqlen: int,
                 mask: torch.tensor, layer_id: int) -> torch.Tensor:
        scores = self.decoding_k(q, k, start_pos, seqlen, layer_id)
        if mask is not None:
            scores[:, :, :, :start_pos+seqlen] += mask
        scores = F.softmax(scores.float(), dim=-1).type_as(q)
        output = self.decoding_v(scores, v, start_pos, seqlen, layer_id)
        output = output.view(q.shape[0], q.shape[1], q.shape[2], q.shape[3])
        return output

    def auto_compute(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int,
                     mask: torch.Tensor = None) -> torch.Tensor:
        seqlen = q.shape[1]
        start_pos = self.get_seq_length(layer_id)
        if mask is not None:
            mask = mask[:, :, :, :start_pos + seqlen]
        if start_pos == 0:
            return self.prefill(q, k, v, seqlen, mask, layer_id)
        else:
            return self.decoding(q, k, v, start_pos, seqlen, mask, layer_id)

    def savekv(self, k: torch.Tensor, v: torch.Tensor, layer_id: int):
        seqlen = k.shape[1]
        start_pos = self.get_seq_length(layer_id)
        self.key_cache[layer_id].save(k, start_pos, seqlen)
        self.value_cache[layer_id].save(v, start_pos, seqlen)

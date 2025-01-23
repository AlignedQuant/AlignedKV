import math
from typing import Optional, Dict, Any, Tuple, List

import torch
from torch import nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from transformers.cache_utils import Cache
from transformers.utils import is_torchdynamo_compiling, logging

from new_extension.util_kcache import KCache, COLUMN_BLOCK_SELF
from new_extension.util_vcache import VCache

logger = logging.get_logger(__name__)


# 注意是所有层共享一个cache
# 在模型generate时直接传入就好
class QuantizedCache_AlignedKV(Cache):
    def __init__(self, config: PretrainedConfig, max_batch_size: int, max_cache_len: int, device, 
                 dtype=None, reference=False, use_tensorcore=False):
        super().__init__()
        max_cache_len = (max_cache_len + COLUMN_BLOCK_SELF - 1) // COLUMN_BLOCK_SELF * COLUMN_BLOCK_SELF
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
        self.num_heads = config.num_attention_heads
        assert self.num_heads % self.num_key_value_heads == 0, f"num_heads {self.num_heads} must be divisible by num_key_value_heads {self.num_key_value_heads}"
        self.gqa = self.num_heads // self.num_key_value_heads

        self.key_cache: List[KCache] = []
        self.value_cache: List[VCache] = []

        for _ in range(config.num_hidden_layers):
            if isinstance(device, dict):
                layer_device = device[_]
            else:
                layer_device = device
            self.key_cache.append(KCache(max_batch_size, self.max_cache_len, self.num_key_value_heads, self.head_dim,
                                         self.gqa, layer_device, _))
            self.value_cache.append(VCache(max_batch_size, self.max_cache_len, self.num_key_value_heads, self.head_dim,
                                           self.gqa, layer_device, _))
        
        self.reference = reference
        self.use_tensorcore = use_tensorcore

        assert dtype is None or dtype == torch.float16, "QuantizedCache_AlignedKV only supports torch.float16"

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
        qlen = self.value_cache[layer_idx].start_pos
        rlen = 0 if self.value_cache[layer_idx].restv is None else self.value_cache[layer_idx].restv.shape[1]
        return qlen + rlen

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states."""
        return self.max_cache_len

    def reset(self):
        """Resets the cache values while preserving the objects"""
        for layer_idx in range(len(self.key_cache)):
            # In-place ops prevent breaking the static address
            self.key_cache[layer_idx].zerolize()
            self.value_cache[layer_idx].zerolize()

    def prefill_save_kv(self, k: torch.Tensor, v: torch.Tensor, layer_id: int):
        self.key_cache[layer_id].prefill_save_k(k)
        self.value_cache[layer_id].prefill_save_v(v)

    def decoding(self, module: nn.Module, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attention_mask: Optional[torch.Tensor],
                 scaling: float, dropout: float, layer_id: int) -> torch.Tensor:
        attn_weights = self.key_cache[layer_id].decoding(q, k, self.reference, self.use_tensorcore)
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : attn_weights.shape[-1]]
            attn_weights = attn_weights + causal_mask
        attn_weights = F.softmax(attn_weights * scaling, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = self.value_cache[layer_id].decoding(attn_weights, v, self.reference, self.use_tensorcore)
        return attn_output, attn_weights

    def is_prefill(self, layer_id: int = 0) -> bool:
        return self.get_seq_length(layer_id) == 0
    
    def prefill_save_o(self, o: torch.Tensor, layer_id: int):
        self.value_cache[layer_id].prefill_o_list(o)
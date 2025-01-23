from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from transformers.utils import (
    logging,
)
from transformers.models.mixtral.configuration_mixtral import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import (
    apply_rotary_pos_emb,
    eager_attention_forward,
    ALL_ATTENTION_FUNCTIONS,
    MixtralAttention,
    MixtralDecoderLayer,
    MixtralModel,
    MixtralForCausalLM,
    MixtralRotaryEmbedding,
    MixtralSparseMoeBlock,
    MixtralRMSNorm,
)

from model.KVCache_AlignedKV import QuantizedCache_AlignedKV

logger = logging.get_logger(__name__)

class MixtralAttention_AlignedKV(MixtralAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: Tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor],
            past_key_value: Optional[Cache] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        device = self.q_proj.weight.device
        hidden_states = hidden_states.to(device)
        attention_mask = attention_mask.to(device) if attention_mask is not None else None
        cache_position = cache_position.to(device) if cache_position is not None else None
        position_embeddings = (pos_emb.to(device) for pos_emb in position_embeddings)

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=2)

        if not isinstance(past_key_value, QuantizedCache_AlignedKV) or past_key_value.is_prefill(self.layer_idx):
            if isinstance(past_key_value, QuantizedCache_AlignedKV):
                past_key_value.prefill_save_kv(key_states, value_states, self.layer_idx)

            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

            if past_key_value is not None and not isinstance(past_key_value, QuantizedCache_AlignedKV):
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

            attention_interface: Callable = eager_attention_forward
            if self.config._attn_implementation != "eager":
                if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                    logger.warning_once(
                        "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                        'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                    )
                else:
                    attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=getattr(self.config, "sliding_window", None),  # main diff with Llama
                **kwargs,
            )

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()

            if isinstance(past_key_value, QuantizedCache_AlignedKV):
                past_key_value.prefill_save_o(attn_output, self.layer_idx)
        else:
            # decoding
            attn_output, attn_weights = past_key_value.decoding(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                self.scaling,
                self.attention_dropout,
                self.layer_idx,
            )
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class MixtralDecoderLayer_AlignedKV(MixtralDecoderLayer):
    def __init__(self, config: MixtralConfig, layer_idx: int):
        super(MixtralDecoderLayer).__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = MixtralAttention_AlignedKV(config, layer_idx)

        self.block_sparse_moe = MixtralSparseMoeBlock(config)
        self.input_layernorm = MixtralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MixtralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class MixtralModel_AlignedKV(MixtralModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MixtralDecoderLayer`]

    Args:
        config: MixtralConfig
    """

    def __init__(self, config: MixtralConfig):
        super(MixtralModel).__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [MixtralDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = MixtralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = MixtralRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()


class MixtralForCausalLM_AlignedKV(MixtralForCausalLM):
    def __init__(self, config):
        super(MixtralForCausalLM).__init__(config)
        self.model = MixtralModel_AlignedKV(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        # Initialize weights and apply final processing
        self.post_init()

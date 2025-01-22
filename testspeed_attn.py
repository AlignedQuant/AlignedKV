import torch
import math
import time
import os
from model.KVCache_AlignedKV import QuantizedCache_AlignedKV
from transformers.cache_utils import DynamicCache, StaticCache
from transformers.models.llama.modeling_llama import eager_attention_forward
from transformers.modeling_utils import sdpa_attention_forward, flash_attention_forward

class EmptyClass:
    pass

layer_idx = 0

def decoding_alignedkv(past_key_value, 
                       query_states, # (bsz, seqlen, n_head, head_dim)
                       key_states, # (bsz, seqlen, n_kv_head, head_dim)
                       value_states, # (bsz, seqlen, n_kv_head, head_dim)
                       attention_mask):
    
    module = EmptyClass()
    module.training = False
    attn_output, attn_weights = past_key_value.decoding(
        module,
        query_states,
        key_states,
        value_states,
        attention_mask,
        1 / math.sqrt(query_states.size(-1)),
        0.0,
        layer_idx
    )

def decoding_normal_result(past_key_value, 
                      query_states, # (bsz, seqlen, n_head, head_dim)
                      key_states, # (bsz, seqlen, n_kv_head, head_dim)
                      value_states, # (bsz, seqlen, n_kv_head, head_dim)
                      attention_mask):
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)
    key_states, value_states = past_key_value.update(
        key_states,
        value_states,
        layer_idx,
        None
    )
    # attention_interface = eager_attention_forward
    attention_interface = sdpa_attention_forward
    # attention_interface = flash_attention_forward
    module = EmptyClass()
    module.training = False
    module.is_causal = True
    module.num_key_value_groups = query_states.size(1) // key_states.size(1)
    # print("query_states", query_states.shape)
    # print("key_states", key_states.shape)
    # print("value_states", value_states.shape)
    attn_output, attn_weights = attention_interface(
        module,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0,
        scaling=1 / math.sqrt(query_states.size(-1))
    )

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

class PretrainedConfig:
    def __init__(self, max_position_embeddings, hidden_size, num_attention_heads, num_hidden_layers, head_dim=None, num_key_value_heads=None):
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads

dir_path = "/home/tanyf/AlignedKV/testvector/llama-3-8b"
# dir_path = "/home/tanyf/AlignedKV/testvector/llama-2-7b"
layer = 0
q_file = f"activation_attention_xq_after_ROPE_layer_{layer}_pos_0_len_16384.pt"
k_file = f"activation_attention_xk_after_ROPE_layer_{layer}_pos_0_len_16384.pt"
v_file = f"activation_attention_xv_layer_{layer}_pos_0_len_16384.pt"
o_file = f"activation_attention_output_before_reshape_layer_{layer}_pos_0_len_16384.pt"
q_path = os.path.join(dir_path, q_file)
k_path = os.path.join(dir_path, k_file)
v_path = os.path.join(dir_path, v_file)
o_path = os.path.join(dir_path, o_file)

def testspeed_alignedkv(start, end, reference=False, use_tensorcore=True):
    q = torch.load(q_path).to(device)
    k = torch.load(k_path).to(device)
    v = torch.load(v_path).to(device)
    o = torch.load(o_path).to(device)

    # q(bsz, seqlen, n_head, head_dim)
    # k(bsz, seqlen, n_kv_head, head_dim)
    # v(bsz, seqlen, n_kv_head, head_dim)
    # o(bsz, n_head, seqlen, head_dim)

    num_key_value_heads = k.size(2)
    config = PretrainedConfig(16384, 4096, 32, 1, 128, num_key_value_heads)
    cache = QuantizedCache_AlignedKV(config, 10, end, device, torch.float16, reference=reference, use_tensorcore=use_tensorcore)

    # prefill
    k_prefill = k[:, 0:start, :, :].contiguous()
    v_prefill = v[:, 0:start, :, :].contiguous()
    o_prefill = o[:, :, 0:start, :].transpose(1, 2).contiguous()
    o_prefill = o_prefill.view(o_prefill.size(0), o_prefill.size(1), -1)
    cache.prefill_save_kv(k_prefill, v_prefill, 0)
    cache.prefill_save_o(o_prefill, 0)

    # decoding
    start_time = time.time()
    for i in range(start, end):
        k_decoding = k[:, i:i+1, :, :].contiguous()
        v_decoding = v[:, i:i+1, :, :].contiguous()
        q_decoding = q[:, i:i+1, :, :].contiguous()
        decoding_alignedkv(cache, q_decoding, k_decoding, v_decoding, None)
        if i % 512 == 0:
            time_cost = time.time() - start_time
            print(f"i={i}, time_cost={time_cost}s")

    end_time = time.time()
    print(f"Total time cost: {end_time - start_time}s")

def testspeed_dynamiccache(start, end):
    q = torch.load(q_path).to(device)
    k = torch.load(k_path).to(device)
    v = torch.load(v_path).to(device)
    o = torch.load(o_path).to(device)

    print(q.shape)
    print(k.shape)
    print(v.shape)
    print(o.shape)

    # q(bsz, seqlen, n_head, head_dim)
    # k(bsz, seqlen, n_kv_head, head_dim)
    # v(bsz, seqlen, n_kv_head, head_dim)
    # o(bsz, n_head, seqlen, head_dim)

    num_key_value_heads = 32
    config = PretrainedConfig(16384, 4096, 32, 1, 128, num_key_value_heads)
    cache = DynamicCache()

    # prefill
    k_prefill = k[:, 0:start, :, :].transpose(1, 2).contiguous()
    v_prefill = v[:, 0:start, :, :].transpose(1, 2).contiguous()
    cache.update(k_prefill, v_prefill, 0, None)

    # decoding
    start_time = time.time()
    for i in range(start, end):
        k_decoding = k[:, i:i+1, :, :].contiguous()
        v_decoding = v[:, i:i+1, :, :].contiguous()
        q_decoding = q[:, i:i+1, :, :].contiguous()
        decoding_normal_result(cache, q_decoding, k_decoding, v_decoding, None)
        if i % 512 == 0:
            time_cost = time.time() - start_time
            print(f"i={i}, time_cost={time_cost}s")

    end_time = time.time()
    print(f"Total time cost: {end_time - start_time}s")

def testspeed_staticcache(start, end):
    q = torch.load(q_path).to(device)
    k = torch.load(k_path).to(device)
    v = torch.load(v_path).to(device)
    o = torch.load(o_path).to(device)

    print(q.shape)
    print(k.shape)
    print(v.shape)
    print(o.shape)

    # q(bsz, seqlen, n_head, head_dim)
    # k(bsz, seqlen, n_kv_head, head_dim)
    # v(bsz, seqlen, n_kv_head, head_dim)
    # o(bsz, n_head, seqlen, head_dim)

    num_key_value_heads = 32
    config = PretrainedConfig(16384, 4096, 32, 1, 128, num_key_value_heads)
    cache = StaticCache(config, 10, end, device, torch.float16)

    # prefill
    k_prefill = k[:, 0:start, :, :].transpose(1, 2).contiguous()
    v_prefill = v[:, 0:start, :, :].transpose(1, 2).contiguous()
    cache.update(k_prefill, v_prefill, 0, None)

    # decoding
    start_time = time.time()
    for i in range(start, end):
        k_decoding = k[:, i:i+1, :, :].contiguous()
        v_decoding = v[:, i:i+1, :, :].contiguous()
        q_decoding = q[:, i:i+1, :, :].contiguous()
        decoding_normal_result(cache, q_decoding, k_decoding, v_decoding, None)
        if i % 512 == 0:
            time_cost = time.time() - start_time
            print(f"i={i}, time_cost={time_cost}s")

    end_time = time.time()
    print(f"Total time cost: {end_time - start_time}s")

with torch.no_grad():
    # testspeed_dynamiccache(4096, 16384)
    testspeed_staticcache(4096, 16384)
    # testspeed_alignedkv(4096, 16384, reference=False, use_tensorcore=False)

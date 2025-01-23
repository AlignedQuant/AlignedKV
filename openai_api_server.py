# https://www.cnblogs.com/apachecn/p/18333049
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from typing import *
from model.Llama_AlignedKV import LlamaForCausalLM_AlignedKV
from model.KVCache_AlignedKV import QuantizedCache_AlignedKV
from new_extension.myextension import COLUMN_BLOCK_SELF

app = FastAPI()

# config
cache_dtype = torch.float16

class Query(BaseModel):
    messages: List[Dict[str, str]]
    model: Optional[str] = None
    request_id: Optional[str] = None
    do_sample: Optional[bool] = None
    stream: Optional[bool] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Optional[List[str]] = None

device_map = {}
cache_map = {}

# Add other model components to cuda:0
device_map["model.embed_tokens.weight"] = "cuda:0"
device_map["model.norm.weight"] = "cuda:0"
device_map["lm_head.weight"] = "cuda:0"

# Loop through layers 0 to 31
for layer in range(32):
    # Define layer range and corresponding device
    if 0 <= layer < 1 or 27 <= layer < 32:
        device = "cuda:0"  # Map to cuda:0
    elif 1 <= layer < 10:
        device = "cuda:1"
    elif 10 <= layer < 19:
        device = "cuda:2"
    elif 19 <= layer < 27:
        device = "cuda:3"
    else:
        continue  # Skip any layers outside the defined range (if needed)

    # Define the layer names for self-attention and MLP components
    device_map[f"model.layers.{layer}.self_attn.q_proj.weight"] = device
    device_map[f"model.layers.{layer}.self_attn.k_proj.weight"] = device
    device_map[f"model.layers.{layer}.self_attn.v_proj.weight"] = device
    device_map[f"model.layers.{layer}.self_attn.o_proj.weight"] = device
    device_map[f"model.layers.{layer}.mlp.gate_proj.weight"] = device
    device_map[f"model.layers.{layer}.mlp.up_proj.weight"] = device
    device_map[f"model.layers.{layer}.mlp.down_proj.weight"] = device
    device_map[f"model.layers.{layer}.input_layernorm.weight"] = device
    device_map[f"model.layers.{layer}.post_attention_layernorm.weight"] = device
    cache_map[layer] = device

model_id = "/home/data/Llama-3.2-3B-Instruct"
model = LlamaForCausalLM_AlignedKV.from_pretrained(model_id, attn_implementation="sdpa", torch_dtype=cache_dtype, device_map=device_map)
tok = AutoTokenizer.from_pretrained(model_id)
model.eval()


@app.post("/chat/completions")
async def generate_response(query: Query):
    iids = tok.apply_chat_template(query.messages, return_tensors="pt").to(device_map["model.embed_tokens.weight"])
    gen_cfg = dict(
        max_new_tokens=query.max_tokens,
        temperature=query.temperature,
        do_sample=query.do_sample,
        top_p=query.top_p,
    )
    max_length = query.max_tokens + iids.shape[-1]
    max_length = (max_length + COLUMN_BLOCK_SELF - 1) // COLUMN_BLOCK_SELF * COLUMN_BLOCK_SELF
    print(f"max_length: {max_length}")
    print(f"iids.shape: {iids.shape}, iid: {iids}")
    KV_Cache = QuantizedCache_AlignedKV(model.config, 1, max_length, cache_map, cache_dtype, reference=False, use_tensorcore=False)
    with torch.no_grad():
        oids = model.generate(
            inputs=iids,
            max_length=max_length,
            past_key_values=KV_Cache, 
            use_cache=True,
            **gen_cfg,
        )
    oids = oids[0][len(iids):-1].tolist()
    output = tok.decode(oids)
    return {
        "choices": [{
            'index': 0, 
            'message': {'role': 'assistant', 'content': output}
        }]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6006)
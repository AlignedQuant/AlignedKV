import torch
from transformers import AutoTokenizer
from model.Llama_AlignedKV import LlamaForCausalLM_AlignedKV
from model.KVCache_AlignedKV import QuantizedCache_AlignedKV

# config
max_batch_size = 1
max_cache_len = 30
cache_dtype = torch.float16

model = LlamaForCausalLM_AlignedKV.from_pretrained("meta-llama/Llama-2-7b-hf", attn_implementation="eager", torch_dtype=cache_dtype)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

KV_Cache = QuantizedCache_AlignedKV(model.config, max_batch_size, max_cache_len, device, cache_dtype)
generate_ids = model.generate(inputs.input_ids, max_length=30, past_key_values=KV_Cache, use_cache=True)

generate_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(generate_text)

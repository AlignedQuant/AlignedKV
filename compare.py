import torch
from transformers import AutoTokenizer
from transformers.cache_utils import StaticCache
from model.Llama_AlignedKV import LlamaForCausalLM_AlignedKV
from model.KVCache_AlignedKV import QuantizedCache_AlignedKV

# config
batch_size = 1
max_len = 128
device = "cuda:0"
cache_dtype = torch.float16
model_path = "./llama2-7b/"

model = LlamaForCausalLM_AlignedKV.from_pretrained(model_path, attn_implementation="eager", torch_dtype=torch.half)
tokenizer = AutoTokenizer.from_pretrained(model_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = "cpu"
model.to(device)

prompt = """Write an TOFEL essay about the following topic: \n What makes a good boss? What are some important qualities of a good supervisor (boss)? \n Use specific details and examples to explain why these qualities are important. \n"""
# prompt = "The presidential order did not restrict U-2 flights outside eastern Europe. In May 1956, Turkey approved the deployment of Detachment B at Incirlik Air Base, near Adana, Turkey. Before the new detachment was ready, however, Detachment A in late August used Adana as a refueling base to photograph the Mediterranean."
inputs = tokenizer(prompt, return_tensors="pt")
inputs = inputs.to(device)

print("token number: ", inputs.input_ids.shape[-1])
print("===================================")

KV_Cache = QuantizedCache_AlignedKV(model.config, batch_size, max_len, device, cache_dtype)
generate_ids = model.generate(inputs.input_ids, max_length=max_len, past_key_values=KV_Cache, use_cache=True, num_beams=1, do_sample=False, temperature=None, top_p=None)

# print(generate_ids)
generate_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(generate_text)
print("===================================")

KV_Cache2 = StaticCache(model.config, batch_size, max_len, device, cache_dtype)
generate_ids2 = model.generate(inputs.input_ids, max_length=max_len, past_key_values=KV_Cache2, use_cache=True, num_beams=1, do_sample=False, temperature=None, top_p=None)
generate_text2 = tokenizer.batch_decode(generate_ids2, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(generate_text2)
print("===================================")
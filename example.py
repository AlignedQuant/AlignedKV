from transformers import AutoTokenizer
from .model.Llama_AlignedKV import LlamaForCausalLM_AlignedKV
from .model.KVCache_AlignedKV import QuantizedCache_AlignedKV

model = LlamaForCausalLM_AlignedKV.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")

cache_kwargs = {
                "config": self.config,
                "max_batch_size": max_batch_size,
                "max_cache_len": max_cache_len,
                "device": device,
                "dtype": cache_dtype,
            }
KV_Cache = QuantizedCache_AlignedKV(**cache_kwargs)
generate_ids = model.generate(inputs.input_ids, max_length=30, past_key_values=KV_Cache, use_cache=True)

generate_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(generate_text)

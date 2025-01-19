import torch
from transformers import AutoTokenizer
from model.Llama_AlignedKV import LlamaForCausalLM_AlignedKV
from model.KVCache_AlignedKV import QuantizedCache_AlignedKV

# config
max_batch_size = 1
max_cache_len = 1024
cache_dtype = torch.float16

model_id = 'meta-llama/Llama-3.1-8B' # "/home/wanghz/GPTQ-triton/model/Meta-Llama-3.1-8B"
model = LlamaForCausalLM_AlignedKV.from_pretrained(model_id, attn_implementation="eager", torch_dtype=cache_dtype)
tokenizer = AutoTokenizer.from_pretrained(model_id)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model.to(device)

prompt = "It is done, and submitted. You can play “Survival of the Tastiest” on Android, and on the web. Playing on the web works, but you have to simulate multi-touch for table moving and that can be a bit confusing.\n\nThere’s a lot I’d like to talk about. I’ll go through every topic, insted of making the typical what went right/wrong list.\n\nConcept\n\nWorking over the theme was probably one of the hardest tasks I had to face.\n\nOriginally, I had an idea of what kind of game I wanted to develop, gameplay wise – something with lots of enemies/actors, simple graphics, maybe set in space, controlled from a top-down view. I was confident I could fit any theme around it.\n\nIn the end, the problem with a theme like “Evolution” in a game is that evolution is unassisted. It happens through several seemingly random mutations over time, with the most apt permutation surviving. This genetic car simulator is, in my opinion, a great example of actual evolution of a species facing a challenge. But is it a game?\n\nIn a game, you need to control something to reach an objective. That control goes against what evolution is supposed to be like. If you allow the user to pick how to evolve something, it’s not evolution anymore – it’s the equivalent of intelligent design, the fable invented by creationists to combat the very idea of evolution. Being agnostic and a Pastafarian, that’s not something that rubbed me the right way.\n\nHence, my biggest dillema when deciding what to create was not with what I wanted to create, but with what I did not. I didn’t want to create an “intelligent design” simulator and wrongly call it evolution.\n\nThis is a problem, of course, every other contestant also had to face. And judging by the entries submitted, not many managed to work around it. I’d say the only real solution was through the use of artificial selection, somehow. So far, I haven’t seen any entry using this at its core gameplay.\n\nAlas, this is just a fun competition and after a while I decided not to be as strict with the game idea, and allowed myself to pick whatever I thought would work out."
inputs = tokenizer(prompt, return_tensors="pt").to(device)
# print(inputs.input_ids.shape)
KV_Cache = QuantizedCache_AlignedKV(model.config, max_batch_size, max_cache_len, device, cache_dtype, reference=False, use_tensorcore=True)
generate_ids = model.generate(inputs.input_ids, max_length=max_cache_len, past_key_values=KV_Cache, use_cache=True)

generate_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(generate_text)

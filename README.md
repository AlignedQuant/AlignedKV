# Evaluate AlignedKV
AlignedKV: Reducing Memory Access of KV-Cache with Precision-Aligned Quantization

## Use Tasks (coqa, truthfulqa_gen, gsm8k) to evaluate the model

Use lm-eval to evaluate model on downstream tasks (e.g. GSM8K, Coqa, etc.):

```bash
cd lm-evaluation-harness
pip install -e .
cd ..
python evaluatetasks.py
```

You can modify `evaluatetasks.py` to select tasks and choose the implementation method for KV-Cache.

```python
task_list = ["coqa", "truthfulqa_gen", "gsm8k"] # "coqa", "truthfulqa_gen", "gsm8k"
device = "cuda:0"
model_path = "meta-llama/Llama-2-7b-hf"
kvcache_type = "alignedKV" # "alignedkv", "static", "kivi"
```

## Compare the outputs of different KV-Cache implementations

```bash
python compare.py
```

You can modify `compare.py` to write an input and wait different KV-Cache implements to generate outputs.

```python
prompt = """Write an TOFEL essay about the following topic: \n What makes a good boss? What are some important qualities of a good supervisor (boss)? \n Use specific details and examples to explain why these qualities are important. \n"""
```

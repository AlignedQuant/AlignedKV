import os
import argparse
import json, tqdm
import torch
import copy

import math
import time
from lm_eval.tasks import TaskManager
from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator, utils, tasks
from lm_eval.utils import handle_non_serializable, make_table, simple_parse_args_string

from model.Llama_AlignedKV import LMEvalLlamaForCausalLM_AlignedKV

# config
# tasks in {coqa, truthfulqa_gen, gsm8k}
task_list = ["coqa", "truthfulqa_gen", "gsm8k"] # "coqa", "truthfulqa_gen", "gsm8k"
device = "cuda:0"
model_path = "meta-llama/Llama-2-7b-hf"
kvcache_type = "alignedKV" # "alignedkv", "static", "kivi"

model = LMEvalLlamaForCausalLM_AlignedKV(
                pretrained=model_path,
                dtype=torch.half,
                max_length=2048,
                batch_size=12,
                device=device,
                attn_implementation="eager",
                key_value_cache_class=kvcache_type,
            )
# model = HFLM(pretrained=model_path, max_length=2048, batch_size=12, device=device)

# tasks.initialize_tasks()
task_manager = TaskManager("INFO", include_path=None)
# print(task_manager.list_all_tasks())
task_names = task_manager.match_tasks(task_list)
for task in [task for task in task_list if task not in task_names]:
    if os.path.isfile(task):
        config = utils.load_yaml_config(task)
        task_names.append(config)
task_missing = [
    task for task in task_list if task not in task_names and "*" not in task
]  # we don't want errors if a wildcard ("*") task name was used
if task_missing:
    missing = ", ".join(task_missing)
    raise ValueError(
        f"Tasks {missing} were not found. Try `lm-eval --tasks list` for list of available tasks."
    )
results = evaluator.simple_evaluate(
    model=model,
    # model_args='parallelize=True',
    tasks=task_names,
    log_samples=True
    # no_cache=True,
    # num_fewshot=data_args.num_fewshot,
)


# dumped = json.dumps(
#     results, indent=2, default=handle_non_serializable, ensure_ascii=False
# )
# print(dumped)

batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))

# evaluation_tracker.save_results_aggregated(
#     results=results, samples=None
# )

# if args.log_samples:
#     for task_name, config in results["configs"].items():
#         evaluation_tracker.save_results_samples(
#             task_name=task_name, samples=samples[task_name]
#         )

# if (
#     evaluation_tracker.push_results_to_hub
#     or evaluation_tracker.push_samples_to_hub
# ):
#     evaluation_tracker.recreate_metadata_card()

print(make_table(results))
if "groups" in results:
    print(make_table(results, "groups"))
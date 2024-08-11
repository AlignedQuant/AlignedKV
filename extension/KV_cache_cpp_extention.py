import torch
from torch.utils.cpp_extension import load
import os
sources_path = [r"./csrc/K_cache.cu", r"./csrc/V_cache.cu", r"./csrc/KV_cache_ops.cpp"]
sources_path = [os.path.join(os.path.dirname(__file__), f) for f in sources_path]
cuda_module = load(name="kv_cache_cuda",
                   sources=sources_path,
                   extra_cuda_cflags=["-O3", "-ccbin=/usr/bin/gcc-10"],
                   verbose=True)
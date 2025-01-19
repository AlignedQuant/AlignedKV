import torch
from torch.utils.cpp_extension import load
import os

COLUMN_BLOCK = 64
COLUMN_BLOCK_SELF = 256

sources_path = [r"./csrc/kcache.cu", r"./csrc/vcache.cu", r"./csrc/bind.cpp"]
sources_path = [os.path.join(os.path.dirname(__file__), path) for path in sources_path]
AlignedKV_extension = load(name="AlignedKV",
                           sources=sources_path,
                           extra_cuda_cflags=["-O3"],
                           verbose=True)
# -*- Python -*-

import os
import subprocess

# Setup config name.
config.name = "objsan"

# Setup source root.
config.test_source_root = os.path.dirname(__file__)

config.suffixes = [".c", ".cpp", ".cu"]

cxx_cuda_args = config.clang + '-x cuda -fsanitize=object -mllvm -objsan-gpu-only=1 -mllvm -objsan-runtime-bitcode -mllvm objsan_ir_rt.bc -fgpu-rdc -foffload-lto --offload-link -Xoffload-linker objsan_rt.o'

config.substitutions.append(
    ('%clangxx_objsan_cuda', cxx_cuda_args))

config.substitutions.append(
    ('%cuda_env', ''))

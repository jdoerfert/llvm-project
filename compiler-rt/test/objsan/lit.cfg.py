# -*- Python -*-

import os
import subprocess

# Setup config name.
config.name = "objsan"

# Setup source root.
config.test_source_root = os.path.dirname(__file__)

config.suffixes = [".c", ".cpp", ".cu"]

# TODO sm_80 need to get from somewhere
cxx_cuda_args = config.clang + f'-x cuda --cuda-path={config.cuda_path} -fsanitize=object -mllvm -objsan-gpu-only=1 -mllvm -objsan-runtime-bitcode -mllvm objsan_ir_rt.bc --offload-arch=sm_80 -fgpu-rdc -foffload-lto --offload-link -Xoffload-linker objsan_rt.o -L{config.cuda_lib_path} -lcudart'

config.substitutions.append(
    ('%clangxx_objsan_cuda', cxx_cuda_args))

config.substitutions.append(
    ('%cuda_preload', config.cuda_preload_path))

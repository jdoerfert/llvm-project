# -*- Python -*-

import os
import subprocess

# TODO not sure if this is a good way to go about getting the cuda gpu arch
def get_cuda_gpu_arch():
  try:
    # Queries the Compute Capability (e.g., "8.0")
    cmd = ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader,nounits']
    arch = subprocess.check_output(cmd).decode('utf-8').strip().split('\n')[0]
    # Convert "8.0" to "sm_80"
    return f"sm_{arch.replace('.', '')}"
  except Exception:
    return None

cuda_gpu_arch = get_cuda_gpu_arch()
if cuda_gpu_arch:
  config.name = "objsan"
  config.test_source_root = os.path.dirname(__file__)
  config.suffixes = [".cuda.cu"]
  
  clang_cuda_compile_args = config.clang + f' -x cuda --cuda-path={config.cuda_path} --offload-arch={cuda_gpu_arch} -fgpu-rdc -foffload-lto'
  clang_cuda_link_args = config.clang + f' --cuda-path={config.cuda_path} --offload-link'
  clang_cuda_post_link_args = f'-L{config.cuda_lib_path} -lcudart -lstdc++'

  config.substitutions.append(
      ('%clang_cuda_compile', clang_cuda_compile_args))
  config.substitutions.append(
      ('%clang_cuda_link', clang_cuda_link_args))
  config.substitutions.append(
      ('%clang_cuda_post_link', clang_cuda_post_link_args))

  clang_objsan_cuda_compile_args = config.clang + f' -x cuda --cuda-path={config.cuda_path} -fsanitize=object -mllvm -objsan-gpu-only=1 --offload-arch={cuda_gpu_arch} -fgpu-rdc -foffload-lto'
  clang_objsan_cuda_link_args = config.clang + f' --cuda-path={config.cuda_path} -fsanitize=object --offload-link -Xoffload-linker {config.cuda_objsan_device_rt}'
  clang_objsan_cuda_post_link_args = f'{config.cuda_preload_path} -L{config.cuda_lib_path} -lcudart -lstdc++'
  
  config.substitutions.append(
      ('%clang_objsan_cuda_compile', clang_objsan_cuda_compile_args))
  config.substitutions.append(
      ('%clang_objsan_cuda_link', clang_objsan_cuda_link_args))
  config.substitutions.append(
      ('%clang_objsan_cuda_post_link', clang_objsan_cuda_post_link_args))
  
  config.substitutions.append(
      ('%cuda_preload', config.cuda_preload_path))

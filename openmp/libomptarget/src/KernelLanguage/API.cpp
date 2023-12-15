
#include "omp.h"
#include <cstdio>

struct dim3 {
  dim3() {}
  dim3(unsigned x) : x(x) {}
  unsigned x = 0, y = 0, z = 0;
};

struct KernelArgsTy {
  uint32_t Version;   // Version of this struct for ABI compatibility.
  uint32_t NumArgs;   // Number of arguments in each input pointer.
  void **ArgBasePtrs; // Base pointer of each argument (e.g. a struct).
  void **ArgPtrs;     // Pointer to the argument data.
  int64_t *ArgSizes;  // Size of the argument data in bytes.
  int64_t *ArgTypes;  // Type of the data (e.g. to / from).
  void **ArgNames;    // Name of the data for debugging, possibly null.
  void **ArgMappers;  // User-defined mappers, possibly null.
  uint64_t Tripcount; // Tripcount for the teams / distribute loop, 0 otherwise.
  struct {
    uint64_t NoWait : 1; // Was this kernel spawned with a `nowait` clause.
    uint64_t Unused : 63;
  } Flags;
  uint32_t NumTeams[3];    // The number of teams (for x,y,z dimension).
  uint32_t ThreadLimit[3]; // The number of threads (for x,y,z dimension).
  uint32_t DynCGroupMem;   // Amount of dynamic cgroup memory requested.
};

struct __omp_kernel_t {
  dim3 __grid_size;
  dim3 __block_size;
  size_t __shared_memory;

  void *__stream;
};

static __omp_kernel_t __current_kernel;
#pragma omp threadprivate(__current_kernel);

extern "C" {

// TODO: There is little reason we need to keep these names or the way calls are
// issued. For now we do to avoid modifying Clang's CUDA codegen.
unsigned __llvmPushCallConfiguration(dim3 __grid_size, dim3 __block_size,
                                     size_t __shared_memory, void *__stream) {
  __omp_kernel_t &__kernel = __current_kernel;
  __kernel.__grid_size = __grid_size;
  __kernel.__block_size = __block_size;
  __kernel.__shared_memory = __shared_memory;
  __kernel.__stream = __stream;
  return 0;
}

unsigned __llvmPopCallConfiguration(dim3 *__grid_size, dim3 *__block_size,
                                    size_t *__shared_memory, void *__stream) {
  __omp_kernel_t &__kernel = __current_kernel;
  *__grid_size = __kernel.__grid_size;
  *__block_size = __kernel.__block_size;
  *__shared_memory = __kernel.__shared_memory;
  *((void **)__stream) = __kernel.__stream;
  return 0;
}

int __tgt_target_kernel(void *Loc, int64_t DeviceId, int32_t NumTeams,
                        int32_t ThreadLimit, const void *HostPtr,
                        KernelArgsTy *Args);

unsigned llvmLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                          void **args, size_t sharedMem, void *stream) {
  // TODO: Determine why we stored the configuration in __current_kernel if we
  // don't need it here.
  //   __omp_kernel_t &__kernel = __current_kernel;

  KernelArgsTy Args = {0};
  Args.DynCGroupMem = sharedMem;
  Args.NumTeams[0] = gridDim.x;
  Args.ThreadLimit[0] = blockDim.x;
  Args.ArgPtrs = args;
  int rv = __tgt_target_kernel(nullptr, omp_get_default_device(), gridDim.x,
                               blockDim.x, func, &Args);
  return rv;
}
}

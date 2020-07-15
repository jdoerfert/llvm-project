// RUN: %clang_cc1                                 -verify=host                                                              -Rpass=openmp -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1                                 -verify=all,safe                                                          -Rpass=openmp -fopenmp -O2 -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o %t.out
// RUN: %clang_cc1 -fexperimental-new-pass-manager -verify=all,safe                                                          -Rpass=openmp -fopenmp -O2 -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o %t.out
// RUN: %clang_cc1                                 -verify=all,force -mllvm -openmp-unsafe-assume-no-external-target-regions -Rpass=openmp -fopenmp -O2 -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o %t.out
// RUN: %clang_cc1 -fexperimental-new-pass-manager -verify=all,force -mllvm -openmp-unsafe-assume-no-external-target-regions -Rpass=openmp -fopenmp -O2 -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o %t.out

// host-no-diagnostics

void bar(void) {
#pragma omp parallel // #1
                     // all-remark@#1 {{Found a parallel region that is called in a target region but not part of a combined target construct nor nesed inside a target construct without intermediate code. This can lead to excessive register usage for unrelated target regions in the same translation unit due to spurious call edges assumed by ptxas.}}
                     // safe-remark@#1 {{Parallel region is not known to be called from a unique single target region, maybe the surrounding function has external linkage?; will not attempt to rewrite the state machine use.}}
                     // force-remark@#1 {{[UNSAFE] Parallel region is not known to be called from a unique single target region, maybe the surrounding function has external linkage?; will rewrite the state machine use due to command line flag, this can lead to undefined behavior if the parallel region is called from a target region outside this translation unit.}}
                     // force-remark@#1 {{Specialize parallel region that is only reached from a single target region to avoid spurious call edges and excessive register usage in other target regions. (parallel region ID: __omp_outlined__2_wrapper, kernel ID: <NONE>}}
  {
  }
}

void foo(void) {
#pragma omp target teams // #2
                         // all-remark@#2 {{Target region containing the parallel region that is specialized. (parallel region ID: __omp_outlined__1_wrapper, kernel ID: __omp_offloading_22}}
                         // all-remark@#2 {{Target region containing the parallel region that is specialized. (parallel region ID: __omp_outlined__3_wrapper, kernel ID: __omp_offloading_22}}
  {
#pragma omp parallel // #3
                     // all-remark@#3 {{Found a parallel region that is called in a target region but not part of a combined target construct nor nesed inside a target construct without intermediate code. This can lead to excessive register usage for unrelated target regions in the same translation unit due to spurious call edges assumed by ptxas.}}
                     // all-remark@#3 {{Specialize parallel region that is only reached from a single target region to avoid spurious call edges and excessive register usage in other target regions. (parallel region ID: __omp_outlined__1_wrapper, kernel ID: __omp_offloading_22}}
    {
    }
    bar();
#pragma omp parallel // #4
                     // all-remark@#4 {{Found a parallel region that is called in a target region but not part of a combined target construct nor nesed inside a target construct without intermediate code. This can lead to excessive register usage for unrelated target regions in the same translation unit due to spurious call edges assumed by ptxas.}}
                     // all-remark@#4 {{Specialize parallel region that is only reached from a single target region to avoid spurious call edges and excessive register usage in other target regions. (parallel region ID: __omp_outlined__3_wrapper, kernel ID: __omp_offloading_22}}
    {
    }
  }
}

void spmd(void) {
  // Verify we do not emit the remarks above for "SPMD" regions.
#pragma omp target teams
#pragma omp parallel
  {
  }

#pragma omp target teams distribute parallel for
  for (int i = 0; i < 100; ++i) {
  }
}

// all-remark@* {{OpenMP runtime call __kmpc_global_thread_num moved to}}
// all-remark@* {{OpenMP runtime call __kmpc_global_thread_num deduplicated}}

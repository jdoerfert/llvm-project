// RUN: %clang_cc1 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -fopenmp -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -mllvm -openmp-tregion-runtime -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s

// CHECK: loop_in_loop_in_tregion
// CHECK:  %0 = call i8 @__kmpc_target_region_kernel_init(%struct.ident_t* null, i1 false, i1 true, i1 true, i1 true)
// CHECK:  call void @__kmpc_target_region_kernel_deinit(%struct.ident_t* null, i1 false, i1 true)
void loop_in_loop_in_tregion(int *A, int *B) {
#pragma omp target
  for (int i = 0; i < 512; i++) {
    for (int j = 0; j < 1024; j++)
      A[j] += B[i + j];
  }
}

// CHECK: parallel_loops_and_accesses_in_tregion
// CHECK:  %0 = call i8 @__kmpc_target_region_kernel_init(%struct.ident_t* null, i1 false, i1 true, i1 true, i1 true)
// CHECK:  call void @__kmpc_target_region_kernel_parallel(%struct.ident_t* null, i1 false, i1 true, void (i8*, i8*)* @.omp_TRegion._wrapper, i8* undef, i16 0, i8* %2, i16 16, i1 false)
// CHECK:  call void @__kmpc_target_region_kernel_parallel(%struct.ident_t* null, i1 false, i1 true, void (i8*, i8*)* @.omp_TRegion.1_wrapper, i8* undef, i16 0, i8* %5, i16 16, i1 false)
// CHECK:  call void @__kmpc_target_region_kernel_parallel(%struct.ident_t* null, i1 false, i1 true, void (i8*, i8*)* @.omp_TRegion.2_wrapper, i8* undef, i16 0, i8* %8, i16 16, i1 false)
// CHECK:  call void @__kmpc_target_region_kernel_deinit(%struct.ident_t* null, i1 false, i1 true)
void parallel_loops_and_accesses_in_tregion(int *A, int *B) {
#pragma omp target
  {
#pragma omp parallel for
    for (int j = 0; j < 1024; j++)
      A[j] += B[0 + j];
#pragma omp parallel for
    for (int j = 0; j < 1024; j++)
      A[j] += B[1 + j];
#pragma omp parallel for
    for (int j = 0; j < 1024; j++)
      A[j] += B[2 + j];

    // This needs a guard in SPMD mode
    A[0] = B[0];
  }
}

void extern_func();
static void parallel_loop(int *A, int *B, int i) {
#pragma omp parallel for
  for (int j = 0; j < 1024; j++)
    A[j] += B[i + j];
}

// CHECK: parallel_loop_in_function_in_loop_with_global_acc_in_tregion
// CHECK:  %1 = call i8 @__kmpc_target_region_kernel_init(%struct.ident_t* null, i1 false, i1 true, i1 true, i1 true)
// CHECK:  call void @__kmpc_target_region_kernel_deinit(%struct.ident_t* null, i1 false, i1 true)
int Global[512];
void parallel_loop_in_function_in_loop_with_global_acc_in_tregion(int *A, int *B) {
#pragma omp target
  for (int i = 0; i < 512; i++) {
    parallel_loop(A, B, i);
    Global[i]++;
  }
}

// CHECK: parallel_loop
// CHECK:  call void @__kmpc_target_region_kernel_parallel(%struct.ident_t* null, i1 false, i1 true, void (i8*, i8*)* @.omp_TRegion.3_wrapper, i8* undef, i16 0, i8* %0, i16 24, i1 false)

// CHECK: parallel_loops_in_functions_and_extern_func_in_tregion
// CHECK:  %0 = call i8 @__kmpc_target_region_kernel_init(%struct.ident_t* null, i1 false, i1 true, i1 true, i1 true)
// CHECK:  call void @__kmpc_target_region_kernel_deinit(%struct.ident_t* null, i1 false, i1 true)
void parallel_loops_in_functions_and_extern_func_in_tregion(int *A, int *B) {
#pragma omp target
  {
    parallel_loop(A, B, 1);
    parallel_loop(A, B, 2);
    extern_func();
    parallel_loop(A, B, 3);
  }
}

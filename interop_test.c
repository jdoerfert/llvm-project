#define CN (1<<11)

#if 0
#define VERBOSE(...) printf(__VA_ARGS__)
#else
#define VERBOSE(...)
#endif

#define USE_OMP_H

#ifdef USE_OMP_H
#include "omp.h"
#endif

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <assert.h>

#ifndef USE_OMP_H
typedef enum omp_interop_property_t {
  omp_interop_type = -1,
  omp_interop_type_name = -2,
  omp_interop_vendor = -3,
  omp_interop_vendor_name = -4,
  omp_interop_device_id = -5,
  omp_interop_device = -6,
  omp_interop_device_context = -7,
  omp_interop_tasksync = -8,
  omp_interop_first = -8
} omp_interop_property_t;
#endif

typedef enum kmp_interop_type_t {
  kmp_interop_type_unknown = -1,
  kmp_interop_type_platform,
  kmp_interop_type_device,
  kmp_interop_type_tasksync,
} kmp_interop_type_t;

typedef struct ident_t ident_t;
typedef struct omp_interop_val_t omp_interop_val_t;
typedef int32_t kmp_int32;
typedef int64_t kmp_int64;
typedef struct kmp_task_t kmp_task_t;
typedef kmp_int32 (*kmp_routine_entry_t)(kmp_int32, void *);

typedef struct kmp_depend_info {
  intptr_t base_addr;
  size_t len;
  struct {
   unsigned in : 1;
   unsigned out : 1;
   unsigned mtx : 1;
  } flags;
} kmp_depend_info_t;

/*extern "C" {*/
void __kmpc_interop_init(ident_t *loc_ref, kmp_int32 gtid,
                         omp_interop_val_t **interop_ptr,
                         kmp_interop_type_t interop_type, kmp_int32 device_id,
                         kmp_int32 ndeps, kmp_depend_info_t *dep_list,
                         kmp_int32 ndeps_noalias,
                         kmp_depend_info_t *noalias_dep_list);

void __kmpc_interop_use(ident_t *loc_ref, kmp_int32 gtid,
                         omp_interop_val_t **interop_ptr,
                         kmp_interop_type_t interop_type, kmp_int32 device_id,
                         kmp_int32 ndeps, kmp_depend_info_t *dep_list,
                         kmp_int32 ndeps_noalias,
                         kmp_depend_info_t *noalias_dep_list);
void __kmpc_interop_destroy(ident_t *loc_ref, kmp_int32 gtid,
                            omp_interop_val_t **interop_ptr,
                            kmp_interop_type_t interop_type,
                            kmp_int32 device_id, kmp_int32 ndeps,
                            kmp_depend_info_t *dep_list,
                            kmp_int32 ndeps_noalias,
                            kmp_depend_info_t *noalias_dep_list);
/*}*/

kmp_task_t *__kmpc_omp_target_task_alloc(ident_t *loc_ref, kmp_int32 gtid,
                                         kmp_int32 flags,
                                         size_t sizeof_kmp_task_t,
                                         size_t sizeof_shareds,
                                         kmp_routine_entry_t task_entry,
                                         kmp_int64 device_id);

#ifdef ASYNC_ONLY
#define __OMP_GET_INTEROP_TY(RETURN_TYPE, SUFFIX)                              \
RETURN_TYPE omp_get_interop_##SUFFIX(omp_interop_val_t **interop_ptr, \
                                            omp_interop_property_t property, \
                                            int *err);
__OMP_GET_INTEROP_TY(intptr_t, int)
__OMP_GET_INTEROP_TY(void *, ptr)
__OMP_GET_INTEROP_TY(const char *, str)
#undef __OMP_GET_INTEROP_TY
#endif

__attribute__((noinline))
void work(double *A, double *B, int N) {
  for (int i =0; i < N; ++i)
    B[i] = i;
  for (int i =0; i < N; ++i)
    for (int j =0; j < N; ++j)
      A[i] += B[j];
}

void busy_work() {
      double *A = (double*)malloc(CN * sizeof(A[0]));
      double *B = (double*)malloc(CN * sizeof(B[0]));
      work(A, B, CN);
      free(A);
      free(B);
}

int __kmpc_global_thread_num(ident_t *);

typedef void* cudaStream_t;

/*extern "C" {*/
int cudaMalloc(void **, size_t);
int cudaMemcpy(void *dst, const void *src, size_t count, int kind);
int cudaMemcpyAsync(void *dst, const void *src, size_t count, int kind, cudaStream_t stream );
int cudaStreamSynchronize(cudaStream_t stream);
/*}*/

// ----------------------------------------------------------------------------------------------
// SKIP AHEAD TILL HERE ;)

int main() {

  /*
   * R0 = cudaMalloc();
   * R1 = cudaMalloc();
   * R2 = 0;
   *
   * Task OMP T0       depend(out:D0) is_device_ptr(R0) nowait
   *   some busy work
   *   R0 = 1;
   *
   *
   * interop init
   *
   * interop use T1    depend( in:D0) depend(out:D2)
   *
   * cudaMemcpyAsync(&R1, &R0, ...)
   *
   * interop destroy   depend( in:D0) depend(out:D2)
   *
   * Task OMP T2                      depend( in:D2) is_device_ptr(R1) map(from:R2) nowait
   *   R2 = R1;
   *
   *
   * After all tasks finished R2 should be 1
   *
   */

  printf("Welcome :)\n#devices %i, device: %i, initial device: %i, default device: %i\n", omp_get_num_devices(), omp_get_device_num(), omp_get_initial_device(), omp_get_default_device());

  int device_id = omp_get_default_device();

  int *R0, *R1, R2 = 0;
  cudaMalloc((void**)&R0, sizeof(int));
  cudaMalloc((void**)&R1, sizeof(int));

  printf("[%lf] Start the parallel master\n", omp_get_wtime());

  #pragma omp parallel master
  {

    int D0, D2;
    kmp_depend_info_t depend_info_T1[2];

    // depend( in:D0)
    depend_info_T1[0].base_addr = (intptr_t)(&D0);
    depend_info_T1[0].len = 4;
    depend_info_T1[0].flags.in = 1;
    depend_info_T1[0].flags.out = 0;
    depend_info_T1[0].flags.mtx = 0;

    // depend(out:D2)
    depend_info_T1[1].base_addr = (intptr_t)(&D2);
    depend_info_T1[1].len = 4;
    depend_info_T1[1].flags.in = 0;
    depend_info_T1[1].flags.out = 1;
    depend_info_T1[1].flags.mtx = 0;

    int gtid= __kmpc_global_thread_num(NULL);
    VERBOSE("Gtid: %i\n", gtid);


    #pragma omp target depend(out:D0) is_device_ptr(R0) nowait
    {
      busy_work();
      busy_work();
      *R0 = 1;
      VERBOSE("[%lf] End OMP T0\n", omp_get_wtime());
    }
    VERBOSE("[%lf] After spawn OMP T0: %i\n", omp_get_wtime(), R2);

    omp_interop_val_t *interop;
    __kmpc_interop_init(NULL, gtid, &interop, kmp_interop_type_tasksync, device_id, 0,
                        NULL, 0, NULL);
    VERBOSE("After interop init: %i\n", R2);

    for (int i = omp_interop_first; i < 0; i++) {
      int err;
      VERBOSE("%i: ", i);
      fflush(stdout);
      long int li = omp_get_interop_int(&interop, (omp_interop_property_t)(i), &err);
      VERBOSE("%15li [E:%3i] ", li, err);
      const void *p = omp_get_interop_ptr(&interop, (omp_interop_property_t)(i), &err);
      VERBOSE("%15p [E:%3i] ", p, err);
      const char *s = omp_get_interop_str(&interop, (omp_interop_property_t)(i), &err);
      VERBOSE("%15s [E:%3i]\n", s, err);
    }
    VERBOSE("After interop inspect: %i\n", R2);


    __kmpc_interop_use(NULL, gtid, &interop, kmp_interop_type_tasksync, device_id, 2,
                       &depend_info_T1[0], 0, NULL);
    VERBOSE("[%lf] After interop use: %i\n", omp_get_wtime(), R2);

    cudaStream_t stream = (omp_get_interop_ptr(&interop, omp_interop_tasksync, NULL));
    VERBOSE("After get stream %p: %i\n", stream, R2);

    int cudaMemcpyDefault = 4;
    printf("Issue CUDA async memcpy       on stream (=%p)\n", stream);
    int r = cudaMemcpyAsync(&R1, &R0, sizeof(int), cudaMemcpyDefault, stream);
    VERBOSE("[%lf] After cuda memcpy async [%i]: %i\n", omp_get_wtime(), r, R2);

    __kmpc_interop_destroy(NULL, gtid, &interop, kmp_interop_type_tasksync, device_id, 2,
                           &depend_info_T1[0], 0, NULL);
    VERBOSE("[%lf] After interop destroy: %i\n", omp_get_wtime(), R2);

    #pragma omp target depend(in:D2) is_device_ptr(R0,R1) map(from:R2) nowait
    {
      R2 = *R1;
      VERBOSE("[%lf] End OMP T2 %i %i\n", omp_get_wtime(), *R0, R2);
    }
    VERBOSE("[%lf] After spawn OMP T2: %i\n", omp_get_wtime(), R2);

    printf("\n[%lf] - Setup done, waiting for T0 to finish, to trigger T1 which triggers T2, result: %i (probably still 0)\n",omp_get_wtime(), R2);

    busy_work();
    busy_work();
    printf("\n[%lf] - After first  wait: result: %i\n", omp_get_wtime(), R2);

    busy_work();
    busy_work();
    printf("\n[%lf] - After second wait: result: %i - Waiting at the parallel region's (implicit) barrier now\n", omp_get_wtime(), R2);
  }

  printf("[%lf] Final: %i (should be 1)\n", omp_get_wtime(), R2);
  assert(R2 == 1);
}

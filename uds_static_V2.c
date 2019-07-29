#include <stdio.h>
#include <math.h>
#include <omp.h>

// TODO: Various parts of the prototype Clang implementation are rather limited
// in scope, error checking, allowed values, etc. This file presents a working
// example based on the original "uds_static_V2". Usage that differs from the
// shown example was **not** tested. Actually, nothing was really tested.

// This is a user-supplied type that the UDS needs to store some information
// and state.  This can be as easy as a single variable (e.g., for a dynamic)
// or something complex such as historic performance data gathered during past
// loop executions.
typedef struct {
  int lb;
  int ub;
  int atomic_ctr;
} loop_record_t;

// Same prototype for all three functions!

int32_t mystatic_init(void * /* ident_t * */ Loc, int32_t ThreadId,
                      int32_t *IsLastIteration,
                      int * /* TODO LB/UB/IV type */ LB, int *UB, int *Stride,
                      void *UserPayload) {
#pragma omp master
  {
    loop_record_t *lr = (loop_record_t *)UserPayload;
    lr->lb = *LB;
    lr->ub = *UB;
    lr->atomic_ctr = *UB;
  }
#pragma omp barrier
  return /*Unsused */ 0;
}

int32_t mystatic_next(void * /* ident_t * */ Loc, int32_t ThreadId,
                      int32_t *IsLastIteration,
                      int * /* TODO LB/UB/IV type */ LB, int *UB, int *Stride,
                      void *UserPayload) {
  loop_record_t *lr = (loop_record_t *)UserPayload;
  if (ThreadId == 0)
    return 0;

#ifdef DEBUG
  #pragma omp critical
  {
  // Thread computes iterations i in [*LB, *UB].
  printf("[%-3i] <%-3i - %-3i | %-3i> [%-3i]\n", ThreadId, lr->lb, lr->ub, *Stride, lr->atomic_ctr);
#endif

  // Here we assign the highest iterations we haven't computed yet.
  // TODO: For now, increasing order of iterations is required.
  // We assign each thread up to "threadID" many iterations.
  *UB = __atomic_fetch_add(&lr->atomic_ctr, -ThreadId - 1, __ATOMIC_SEQ_CST);
  *LB = *UB - ThreadId;
  *LB = *LB < 0 ? 0 : *LB;

#ifdef DEBUG
  printf("[%-3i] <%-3i - %-3i | %-3i> [%-3i]\n", ThreadId, *LB, *UB, *Stride, lr->atomic_ctr);
  }
#endif

  // TODO: for lastprivate, maybe off by one, e.g., *UB + 1 == lr->ub
  *IsLastIteration = *UB == lr->ub;

  // Only if the return value is true (!= 0) the thread will execute the body.
  return *LB <= *UB;
}

int32_t mystatic_fini(void * /* ident_t * */ Loc, int32_t ThreadId,
                      int32_t *IsLastIteration,
                      int * /* TODO LB/UB/IV type */ LB, int *UB, int *Stride,
                      void *UserPayload) {
  return /*Unsused */ 0;
}

// The schedule requires all 4 entries for now. One could "easily" allow to
// partial definitions, overwrites, or other features. The following example
// shows how an extensions could look like. Note that all functions shall exist
// etc.
//
// #pragma omp declare schedule(mytemplate_partial, init = mytemplate_init, \
//                              fini = mytemplate_fini)
// #pragma omp declare schedule(mytemplate_w_default, init = mytemplate_init, \
//                              next = default_next, fini = mytemplate_fini)
//
// void foo() {
//  #pramga omp declare schedule(mytemplate_partial, next=local_next)
//  #pragma omp parallel for schedule(user:mytemplate_partial, payload)
//      for (int i = 0; i < sz; i++) { ... }
//
//  #pramga omp declare schedule(mytemplate_w_default, next=local_next)
//  #pragma omp parallel for schedule(user:mytemplate_w_default, payload)
//      for (int i = 0; i < sz; i++) { ... }
// }

#pragma omp declare schedule(mystatic, init = mystatic_init,                   \
                             next = mystatic_next, fini = mystatic_fini)

void example(int * array, int sz) {
    loop_record_t lr;
#pragma omp parallel for schedule(user:mystatic, &lr)
    for (int i = 0; i < sz; i++) {
        // this is just to determine if all iterations have been computed
#pragma omp atomic
        array[i]++;
    }
}


void check_array(int * array, int sz) {
    int wrong = 0;
    for (int i = 0; i < sz; i++) {
        if (array[i] != 1) {
            fprintf(stderr, "array[%d]=%d! WRONG!\n", i, array[i]);
            wrong++;
        }
    }
    fprintf(stderr, "%d elements were wrong!\n", wrong);
}

//void dump_array(int * array, int sz) {
//    for (int i = 0; i < sz; ++i) {
//        printf("%d ", array[i]);
//        if ((i+1) % 16 == 0) {
//            printf("\n");
//        }
//    }
//    printf("\n");
//}

#define N 22
int main(int argc, char * argv[]) {
    int array[N];

    for (int i = 0; i < N; i++) {
        array[i] = 0;
    }

    example(array, N);
    check_array(array, N);
    // dump_array(array, N);

    return 0;
}

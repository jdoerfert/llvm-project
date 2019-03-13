; RUN: opt < %s -openmp-opt -stats -disable-output 2>&1 | FileCheck %s --check-prefix=STATS
; RUN: opt < %s -openmp-opt -S 2>&1 | FileCheck %s
;
; REQUIRES: asserts
;
; Check that we will not execute any of the below target regions in SPMD-mode.
; TODO: SPMD-mode is valid for target region 2 and 3 if proper guarding code is inserted.
;
; See the to_SPMD_mode.ll file for almost the same functions that can be translated to SPMD mode.
;
; STATS-DAG: 1 openmp-opt - Number of GPU kernel in non-SPMD mode without parallelism
; STATS-DAG: 3 openmp-opt - Number of custom GPU kernel non-SPMD mode state machines created
; STATS-DAG: 2 openmp-opt - Number of custom GPU kernel non-SPMD mode state machines without fallback
;
; No state machine needed because there is no parallel region.
; CHECK: void @{{.*}}loop_in_loop_in_tregion
; CHECK: call i8 @__kmpc_target_region_kernel_init(%struct.ident_t* @{{[a-zA-Z0-9]*}}, i1 false, i1 {{[a-z]*}}, i1 true
; CHECK: call void @__kmpc_target_region_kernel_deinit(%struct.ident_t* @{{[a-zA-Z0-9]*}}, i1 false,
;
; void loop_in_loop_in_tregion(int *A, int *B) {
; #pragma omp target
;   for (int i = 0; i < 512; i++) {
;     for (int j = 0; j < 1024; j++)
;       A[j] += B[i + j];
;   }
; }
;
;
; Custom state machine needed but no fallback because all parallel regions are known
; CHECK: void @{{.*}}parallel_loops_and_accesses_in_tregion
; CHECK: call i8 @__kmpc_target_region_kernel_init(%struct.ident_t* @{{[a-zA-Z0-9]*}}, i1 false, i1 {{[a-z]*}}, i1 false
; The "check.next" block should not contain a fallback call
; CHECK:       worker.check.next4:
; CHECK-NEXT:    br label %worker.parallel_end
; CHECK: call void @__kmpc_target_region_kernel_deinit(%struct.ident_t* @{{[a-zA-Z0-9]*}}, i1 false,
;
; void parallel_loops_and_accesses_in_tregion(int *A, int *B) {
; #pragma omp target
;   {
; #pragma omp parallel for
;     for (int j = 0; j < 1024; j++)
;       A[j] += B[0 + j];
; #pragma omp parallel for
;     for (int j = 0; j < 1024; j++)
;       A[j] += B[1 + j];
; #pragma omp parallel for
;     for (int j = 0; j < 1024; j++)
;       A[j] += B[2 + j];
;
;     // This needs a guard in SPMD mode
;     A[0] = B[0];
;   }
; }
;
; void extern_func();
; static void parallel_loop(int *A, int *B, int i) {
; #pragma omp parallel for
;   for (int j = 0; j < 1024; j++)
;     A[j] += B[i + j];
; }
;
; int Global[512];
;
;
; Custom state machine needed but no fallback because all parallel regions are known
; CHECK: void @{{.*}}parallel_loop_in_function_in_loop_with_global_acc_in_tregion
; CHECK: call i8 @__kmpc_target_region_kernel_init(%struct.ident_t* @{{[a-zA-Z0-9]*}}, i1 false, i1 {{[a-z]*}}, i1 false
; The "check.next" block should not contain a fallback call
; CHECK:       worker.check.next:
; CHECK-NEXT:    br label %worker.parallel_end
; CHECK: call void @__kmpc_target_region_kernel_deinit(%struct.ident_t* @{{[a-zA-Z0-9]*}}, i1 false,
;
; void parallel_loop_in_function_in_loop_with_global_acc_in_tregion(int *A, int *B) {
; #pragma omp target
;   for (int i = 0; i < 512; i++) {
;     parallel_loop(A, B, i);
;     Global[i]++;
;   }
; }
;
; Custom state machine needed with fallback because "extern_func" might contain parallel regions.
; CHECK: void @{{.*}}parallel_loops_in_functions_and_extern_func_in_tregion
; CHECK: call i8 @__kmpc_target_region_kernel_init(%struct.ident_t* @{{[a-zA-Z0-9]*}}, i1 false, i1 {{[a-z]*}}, i1 false
; The "check.next" block should contain a fallback call
; CHECK:       worker.check.next:
; CHECK-NEXT:    call void %work_fn(
; CHECK-NEXT:    br label %worker.parallel_end
; CHECK: call void @__kmpc_target_region_kernel_deinit(%struct.ident_t* @{{[a-zA-Z0-9]*}}, i1 false,
;
; void parallel_loops_in_functions_and_extern_func_in_tregion(int *A, int *B) {
; #pragma omp target
;   {
;     parallel_loop(A, B, 1);
;     parallel_loop(A, B, 2);
;     extern_func();
;     parallel_loop(A, B, 3);
;   }
; }

source_filename = "../llvm/test/Transforms/OpenMP/no_SPMD_mode.c"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvida-cuda"

%struct.ident_t = type { i32, i32, i32, i32, i8* }
%omp.private.struct = type { i32**, i32** }
%omp.private.struct.0 = type { i32**, i32** }
%omp.private.struct.1 = type { i32**, i32** }
%omp.private.struct.2 = type { i32**, i32**, i32* }

@__omp_offloading_18_29b03e4_loop_in_loop_in_tregion_l2_exec_mode = weak constant i8 1
@.str = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@0 = private unnamed_addr global %struct.ident_t { i32 0, i32 514, i32 0, i32 0, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str, i32 0, i32 0) }, align 8
@1 = private unnamed_addr global %struct.ident_t { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str, i32 0, i32 0) }, align 8
@__omp_offloading_18_29b03e4_parallel_loops_and_accesses_in_tregion_l9_exec_mode = weak constant i8 1
@__omp_offloading_18_29b03e4_parallel_loop_in_function_in_loop_with_global_acc_in_tregion_l35_exec_mode = weak constant i8 1
@__omp_offloading_18_29b03e4_parallel_loops_in_functions_and_extern_func_in_tregion_l43_exec_mode = weak constant i8 1
@llvm.compiler.used = appending global [4 x i8*] [i8* @__omp_offloading_18_29b03e4_loop_in_loop_in_tregion_l2_exec_mode, i8* @__omp_offloading_18_29b03e4_parallel_loops_and_accesses_in_tregion_l9_exec_mode, i8* @__omp_offloading_18_29b03e4_parallel_loop_in_function_in_loop_with_global_acc_in_tregion_l35_exec_mode, i8* @__omp_offloading_18_29b03e4_parallel_loops_in_functions_and_extern_func_in_tregion_l43_exec_mode], section "llvm.metadata"

; Function Attrs: norecurse nounwind
define weak void @__omp_offloading_18_29b03e4_loop_in_loop_in_tregion_l2(i32* %A, i32* %B) #0 {
entry:
  %A.addr = alloca i32*, align 8
  %B.addr = alloca i32*, align 8
  %i = alloca i32, align 4
  %cleanup.dest.slot = alloca i32, align 4
  %j = alloca i32, align 4
  store i32* %A, i32** %A.addr, align 8, !tbaa !11
  store i32* %B, i32** %B.addr, align 8, !tbaa !11
  %0 = call i8 @__kmpc_target_region_kernel_init(%struct.ident_t* @0, i1 false, i1 true, i1 true, i1 true)
  %1 = icmp eq i8 %0, 1
  br i1 %1, label %.execute, label %.exit

.execute:                                         ; preds = %entry
  %2 = bitcast i32* %i to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %2) #4
  store i32 0, i32* %i, align 4, !tbaa !15
  br label %for.cond

for.cond:                                         ; preds = %for.inc8, %.execute
  %3 = load i32, i32* %i, align 4, !tbaa !15
  %cmp = icmp slt i32 %3, 512
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  store i32 2, i32* %cleanup.dest.slot, align 4
  %4 = bitcast i32* %i to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %4) #4
  br label %for.end10

for.body:                                         ; preds = %for.cond
  %5 = bitcast i32* %j to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %5) #4
  store i32 0, i32* %j, align 4, !tbaa !15
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %for.body
  %6 = load i32, i32* %j, align 4, !tbaa !15
  %cmp2 = icmp slt i32 %6, 1024
  br i1 %cmp2, label %for.body4, label %for.cond.cleanup3

for.cond.cleanup3:                                ; preds = %for.cond1
  store i32 5, i32* %cleanup.dest.slot, align 4
  %7 = bitcast i32* %j to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %7) #4
  br label %for.end

for.body4:                                        ; preds = %for.cond1
  %8 = load i32*, i32** %B.addr, align 8, !tbaa !11
  %9 = load i32, i32* %i, align 4, !tbaa !15
  %10 = load i32, i32* %j, align 4, !tbaa !15
  %add = add nsw i32 %9, %10
  %idxprom = sext i32 %add to i64
  %arrayidx = getelementptr inbounds i32, i32* %8, i64 %idxprom
  %11 = load i32, i32* %arrayidx, align 4, !tbaa !15
  %12 = load i32*, i32** %A.addr, align 8, !tbaa !11
  %13 = load i32, i32* %j, align 4, !tbaa !15
  %idxprom5 = sext i32 %13 to i64
  %arrayidx6 = getelementptr inbounds i32, i32* %12, i64 %idxprom5
  %14 = load i32, i32* %arrayidx6, align 4, !tbaa !15
  %add7 = add nsw i32 %14, %11
  store i32 %add7, i32* %arrayidx6, align 4, !tbaa !15
  br label %for.inc

for.inc:                                          ; preds = %for.body4
  %15 = load i32, i32* %j, align 4, !tbaa !15
  %inc = add nsw i32 %15, 1
  store i32 %inc, i32* %j, align 4, !tbaa !15
  br label %for.cond1

for.end:                                          ; preds = %for.cond.cleanup3
  br label %for.inc8

for.inc8:                                         ; preds = %for.end
  %16 = load i32, i32* %i, align 4, !tbaa !15
  %inc9 = add nsw i32 %16, 1
  store i32 %inc9, i32* %i, align 4, !tbaa !15
  br label %for.cond

for.end10:                                        ; preds = %for.cond.cleanup
  br label %.omp.deinit

.omp.deinit:                                      ; preds = %for.end10
  call void @__kmpc_target_region_kernel_deinit(%struct.ident_t* @0, i1 false, i1 true)
  br label %.exit

.exit:                                            ; preds = %.omp.deinit, %entry
  ret void
}

declare i8 @__kmpc_target_region_kernel_init(%struct.ident_t*, i1, i1, i1, i1)

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

declare void @__kmpc_target_region_kernel_deinit(%struct.ident_t*, i1, i1)

; Function Attrs: norecurse nounwind
define weak void @__omp_offloading_18_29b03e4_parallel_loops_and_accesses_in_tregion_l9(i32* %A, i32* %B) #0 {
entry:
  %A.addr = alloca i32*, align 8
  %B.addr = alloca i32*, align 8
  %.private.vars = alloca %omp.private.struct, align 8
  %.private.vars1 = alloca %omp.private.struct.0, align 8
  %.private.vars2 = alloca %omp.private.struct.1, align 8
  store i32* %A, i32** %A.addr, align 8, !tbaa !11
  store i32* %B, i32** %B.addr, align 8, !tbaa !11
  %0 = call i8 @__kmpc_target_region_kernel_init(%struct.ident_t* @0, i1 false, i1 true, i1 true, i1 true)
  %1 = icmp eq i8 %0, 1
  br i1 %1, label %.execute, label %.exit

.execute:                                         ; preds = %entry
  %2 = bitcast %omp.private.struct* %.private.vars to i8*
  %3 = getelementptr inbounds %omp.private.struct, %omp.private.struct* %.private.vars, i32 0, i32 0
  store i32** %A.addr, i32*** %3
  %4 = getelementptr inbounds %omp.private.struct, %omp.private.struct* %.private.vars, i32 0, i32 1
  store i32** %B.addr, i32*** %4
  call void @__kmpc_target_region_kernel_parallel(%struct.ident_t* @0, i16 -1, i1 true, void (i8*, i8*)* @.omp_TRegion._wrapper, i8* undef, i16 0, i8* %2, i16 16, i1 false)
  %5 = bitcast %omp.private.struct.0* %.private.vars1 to i8*
  %6 = getelementptr inbounds %omp.private.struct.0, %omp.private.struct.0* %.private.vars1, i32 0, i32 0
  store i32** %A.addr, i32*** %6
  %7 = getelementptr inbounds %omp.private.struct.0, %omp.private.struct.0* %.private.vars1, i32 0, i32 1
  store i32** %B.addr, i32*** %7
  call void @__kmpc_target_region_kernel_parallel(%struct.ident_t* @0, i16 -1, i1 true, void (i8*, i8*)* @.omp_TRegion.1_wrapper, i8* undef, i16 0, i8* %5, i16 16, i1 false)
  %8 = bitcast %omp.private.struct.1* %.private.vars2 to i8*
  %9 = getelementptr inbounds %omp.private.struct.1, %omp.private.struct.1* %.private.vars2, i32 0, i32 0
  store i32** %A.addr, i32*** %9
  %10 = getelementptr inbounds %omp.private.struct.1, %omp.private.struct.1* %.private.vars2, i32 0, i32 1
  store i32** %B.addr, i32*** %10
  call void @__kmpc_target_region_kernel_parallel(%struct.ident_t* @0, i16 -1, i1 true, void (i8*, i8*)* @.omp_TRegion.2_wrapper, i8* undef, i16 0, i8* %8, i16 16, i1 false)
  %11 = load i32*, i32** %B.addr, align 8, !tbaa !11
  %arrayidx = getelementptr inbounds i32, i32* %11, i64 0
  %12 = load i32, i32* %arrayidx, align 4, !tbaa !15
  %13 = load i32*, i32** %A.addr, align 8, !tbaa !11
  %arrayidx3 = getelementptr inbounds i32, i32* %13, i64 0
  store i32 %12, i32* %arrayidx3, align 4, !tbaa !15
  br label %.omp.deinit

.omp.deinit:                                      ; preds = %.execute
  call void @__kmpc_target_region_kernel_deinit(%struct.ident_t* @0, i1 false, i1 true)
  br label %.exit

.exit:                                            ; preds = %.omp.deinit, %entry
  ret void
}

; Function Attrs: norecurse nounwind
define internal void @.omp_TRegion.(i32* noalias %.global_tid., i32* noalias %.bound_tid., i32** dereferenceable(8) %A, i32** dereferenceable(8) %B) #0 {
entry:
  %.global_tid..addr = alloca i32*, align 8
  %.bound_tid..addr = alloca i32*, align 8
  %A.addr = alloca i32**, align 8
  %B.addr = alloca i32**, align 8
  %.omp.iv = alloca i32, align 4
  %tmp = alloca i32, align 4
  %.omp.lb = alloca i32, align 4
  %.omp.ub = alloca i32, align 4
  %.omp.stride = alloca i32, align 4
  %.omp.is_last = alloca i32, align 4
  %j = alloca i32, align 4
  store i32* %.global_tid., i32** %.global_tid..addr, align 8, !tbaa !11
  store i32* %.bound_tid., i32** %.bound_tid..addr, align 8, !tbaa !11
  store i32** %A, i32*** %A.addr, align 8, !tbaa !11
  store i32** %B, i32*** %B.addr, align 8, !tbaa !11
  %0 = load i32**, i32*** %A.addr, align 8, !tbaa !11
  %1 = load i32**, i32*** %B.addr, align 8, !tbaa !11
  %2 = bitcast i32* %.omp.iv to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %2) #4
  %3 = bitcast i32* %.omp.lb to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %3) #4
  store i32 0, i32* %.omp.lb, align 4, !tbaa !15
  %4 = bitcast i32* %.omp.ub to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %4) #4
  store i32 1023, i32* %.omp.ub, align 4, !tbaa !15
  %5 = bitcast i32* %.omp.stride to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %5) #4
  store i32 1, i32* %.omp.stride, align 4, !tbaa !15
  %6 = bitcast i32* %.omp.is_last to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %6) #4
  store i32 0, i32* %.omp.is_last, align 4, !tbaa !15
  %7 = bitcast i32* %j to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %7) #4
  %8 = load i32*, i32** %.global_tid..addr, align 8
  %9 = load i32, i32* %8, align 4, !tbaa !15
  call void @__kmpc_for_static_init_4(%struct.ident_t* @0, i32 %9, i32 33, i32* %.omp.is_last, i32* %.omp.lb, i32* %.omp.ub, i32* %.omp.stride, i32 1, i32 1)
  br label %omp.dispatch.cond

omp.dispatch.cond:                                ; preds = %omp.dispatch.inc, %entry
  %10 = load i32, i32* %.omp.ub, align 4, !tbaa !15
  %cmp = icmp sgt i32 %10, 1023
  br i1 %cmp, label %cond.true, label %cond.false

cond.true:                                        ; preds = %omp.dispatch.cond
  br label %cond.end

cond.false:                                       ; preds = %omp.dispatch.cond
  %11 = load i32, i32* %.omp.ub, align 4, !tbaa !15
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi i32 [ 1023, %cond.true ], [ %11, %cond.false ]
  store i32 %cond, i32* %.omp.ub, align 4, !tbaa !15
  %12 = load i32, i32* %.omp.lb, align 4, !tbaa !15
  store i32 %12, i32* %.omp.iv, align 4, !tbaa !15
  %13 = load i32, i32* %.omp.iv, align 4, !tbaa !15
  %14 = load i32, i32* %.omp.ub, align 4, !tbaa !15
  %cmp1 = icmp sle i32 %13, %14
  br i1 %cmp1, label %omp.dispatch.body, label %omp.dispatch.cleanup

omp.dispatch.cleanup:                             ; preds = %cond.end
  br label %omp.dispatch.end

omp.dispatch.body:                                ; preds = %cond.end
  br label %omp.inner.for.cond

omp.inner.for.cond:                               ; preds = %omp.inner.for.inc, %omp.dispatch.body
  %15 = load i32, i32* %.omp.iv, align 4, !tbaa !15
  %16 = load i32, i32* %.omp.ub, align 4, !tbaa !15
  %cmp2 = icmp sle i32 %15, %16
  br i1 %cmp2, label %omp.inner.for.body, label %omp.inner.for.cond.cleanup

omp.inner.for.cond.cleanup:                       ; preds = %omp.inner.for.cond
  br label %omp.inner.for.end

omp.inner.for.body:                               ; preds = %omp.inner.for.cond
  %17 = load i32, i32* %.omp.iv, align 4, !tbaa !15
  %mul = mul nsw i32 %17, 1
  %add = add nsw i32 0, %mul
  store i32 %add, i32* %j, align 4, !tbaa !15
  %18 = load i32*, i32** %1, align 8, !tbaa !11
  %19 = load i32, i32* %j, align 4, !tbaa !15
  %add3 = add nsw i32 0, %19
  %idxprom = sext i32 %add3 to i64
  %arrayidx = getelementptr inbounds i32, i32* %18, i64 %idxprom
  %20 = load i32, i32* %arrayidx, align 4, !tbaa !15
  %21 = load i32*, i32** %0, align 8, !tbaa !11
  %22 = load i32, i32* %j, align 4, !tbaa !15
  %idxprom4 = sext i32 %22 to i64
  %arrayidx5 = getelementptr inbounds i32, i32* %21, i64 %idxprom4
  %23 = load i32, i32* %arrayidx5, align 4, !tbaa !15
  %add6 = add nsw i32 %23, %20
  store i32 %add6, i32* %arrayidx5, align 4, !tbaa !15
  br label %omp.body.continue

omp.body.continue:                                ; preds = %omp.inner.for.body
  br label %omp.inner.for.inc

omp.inner.for.inc:                                ; preds = %omp.body.continue
  %24 = load i32, i32* %.omp.iv, align 4, !tbaa !15
  %add7 = add nsw i32 %24, 1
  store i32 %add7, i32* %.omp.iv, align 4, !tbaa !15
  br label %omp.inner.for.cond

omp.inner.for.end:                                ; preds = %omp.inner.for.cond.cleanup
  br label %omp.dispatch.inc

omp.dispatch.inc:                                 ; preds = %omp.inner.for.end
  %25 = load i32, i32* %.omp.lb, align 4, !tbaa !15
  %26 = load i32, i32* %.omp.stride, align 4, !tbaa !15
  %add8 = add nsw i32 %25, %26
  store i32 %add8, i32* %.omp.lb, align 4, !tbaa !15
  %27 = load i32, i32* %.omp.ub, align 4, !tbaa !15
  %28 = load i32, i32* %.omp.stride, align 4, !tbaa !15
  %add9 = add nsw i32 %27, %28
  store i32 %add9, i32* %.omp.ub, align 4, !tbaa !15
  br label %omp.dispatch.cond

omp.dispatch.end:                                 ; preds = %omp.dispatch.cleanup
  call void @__kmpc_for_static_fini(%struct.ident_t* @0, i32 %9)
  %29 = bitcast i32* %j to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %29) #4
  %30 = bitcast i32* %.omp.is_last to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %30) #4
  %31 = bitcast i32* %.omp.stride to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %31) #4
  %32 = bitcast i32* %.omp.ub to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %32) #4
  %33 = bitcast i32* %.omp.lb to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %33) #4
  %34 = bitcast i32* %.omp.iv to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %34) #4
  ret void
}

declare void @__kmpc_for_static_init_4(%struct.ident_t*, i32, i32, i32*, i32*, i32*, i32*, i32, i32)

declare void @__kmpc_for_static_fini(%struct.ident_t*, i32)

; Function Attrs: norecurse nounwind
define internal void @.omp_TRegion._wrapper(i8* %shared_vars, i8* %private_vars) #0 {
entry:
  %.addr = alloca i8*, align 8
  %.addr1 = alloca i8*, align 8
  %.zero.addr = alloca i32, align 4
  %.threadid_temp. = alloca i32, align 4
  store i32 0, i32* %.zero.addr, align 4
  store i8* %shared_vars, i8** %.addr, align 8, !tbaa !11
  store i8* %private_vars, i8** %.addr1, align 8, !tbaa !11
  %0 = call i32 @__kmpc_global_thread_num(%struct.ident_t* @1)
  store i32 %0, i32* %.threadid_temp., align 4, !tbaa !15
  %1 = bitcast i8* %private_vars to %omp.private.struct*
  %2 = getelementptr inbounds %omp.private.struct, %omp.private.struct* %1, i32 0, i32 0
  %3 = load i32**, i32*** %2, align 1
  %4 = getelementptr inbounds %omp.private.struct, %omp.private.struct* %1, i32 0, i32 1
  %5 = load i32**, i32*** %4, align 1
  call void @.omp_TRegion.(i32* %.threadid_temp., i32* %.zero.addr, i32** %3, i32** %5) #4
  ret void
}

declare i32 @__kmpc_global_thread_num(%struct.ident_t*)

declare !callback !17 void @__kmpc_target_region_kernel_parallel(%struct.ident_t*, i16, i1, void (i8*, i8*)* nocapture, i8* nocapture, i16, i8* nocapture readonly, i16, i1)

; Function Attrs: norecurse nounwind
define internal void @.omp_TRegion.1(i32* noalias %.global_tid., i32* noalias %.bound_tid., i32** dereferenceable(8) %A, i32** dereferenceable(8) %B) #0 {
entry:
  %.global_tid..addr = alloca i32*, align 8
  %.bound_tid..addr = alloca i32*, align 8
  %A.addr = alloca i32**, align 8
  %B.addr = alloca i32**, align 8
  %.omp.iv = alloca i32, align 4
  %tmp = alloca i32, align 4
  %.omp.lb = alloca i32, align 4
  %.omp.ub = alloca i32, align 4
  %.omp.stride = alloca i32, align 4
  %.omp.is_last = alloca i32, align 4
  %j = alloca i32, align 4
  store i32* %.global_tid., i32** %.global_tid..addr, align 8, !tbaa !11
  store i32* %.bound_tid., i32** %.bound_tid..addr, align 8, !tbaa !11
  store i32** %A, i32*** %A.addr, align 8, !tbaa !11
  store i32** %B, i32*** %B.addr, align 8, !tbaa !11
  %0 = load i32**, i32*** %A.addr, align 8, !tbaa !11
  %1 = load i32**, i32*** %B.addr, align 8, !tbaa !11
  %2 = bitcast i32* %.omp.iv to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %2) #4
  %3 = bitcast i32* %.omp.lb to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %3) #4
  store i32 0, i32* %.omp.lb, align 4, !tbaa !15
  %4 = bitcast i32* %.omp.ub to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %4) #4
  store i32 1023, i32* %.omp.ub, align 4, !tbaa !15
  %5 = bitcast i32* %.omp.stride to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %5) #4
  store i32 1, i32* %.omp.stride, align 4, !tbaa !15
  %6 = bitcast i32* %.omp.is_last to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %6) #4
  store i32 0, i32* %.omp.is_last, align 4, !tbaa !15
  %7 = bitcast i32* %j to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %7) #4
  %8 = load i32*, i32** %.global_tid..addr, align 8
  %9 = load i32, i32* %8, align 4, !tbaa !15
  call void @__kmpc_for_static_init_4(%struct.ident_t* @0, i32 %9, i32 33, i32* %.omp.is_last, i32* %.omp.lb, i32* %.omp.ub, i32* %.omp.stride, i32 1, i32 1)
  br label %omp.dispatch.cond

omp.dispatch.cond:                                ; preds = %omp.dispatch.inc, %entry
  %10 = load i32, i32* %.omp.ub, align 4, !tbaa !15
  %cmp = icmp sgt i32 %10, 1023
  br i1 %cmp, label %cond.true, label %cond.false

cond.true:                                        ; preds = %omp.dispatch.cond
  br label %cond.end

cond.false:                                       ; preds = %omp.dispatch.cond
  %11 = load i32, i32* %.omp.ub, align 4, !tbaa !15
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi i32 [ 1023, %cond.true ], [ %11, %cond.false ]
  store i32 %cond, i32* %.omp.ub, align 4, !tbaa !15
  %12 = load i32, i32* %.omp.lb, align 4, !tbaa !15
  store i32 %12, i32* %.omp.iv, align 4, !tbaa !15
  %13 = load i32, i32* %.omp.iv, align 4, !tbaa !15
  %14 = load i32, i32* %.omp.ub, align 4, !tbaa !15
  %cmp1 = icmp sle i32 %13, %14
  br i1 %cmp1, label %omp.dispatch.body, label %omp.dispatch.cleanup

omp.dispatch.cleanup:                             ; preds = %cond.end
  br label %omp.dispatch.end

omp.dispatch.body:                                ; preds = %cond.end
  br label %omp.inner.for.cond

omp.inner.for.cond:                               ; preds = %omp.inner.for.inc, %omp.dispatch.body
  %15 = load i32, i32* %.omp.iv, align 4, !tbaa !15
  %16 = load i32, i32* %.omp.ub, align 4, !tbaa !15
  %cmp2 = icmp sle i32 %15, %16
  br i1 %cmp2, label %omp.inner.for.body, label %omp.inner.for.cond.cleanup

omp.inner.for.cond.cleanup:                       ; preds = %omp.inner.for.cond
  br label %omp.inner.for.end

omp.inner.for.body:                               ; preds = %omp.inner.for.cond
  %17 = load i32, i32* %.omp.iv, align 4, !tbaa !15
  %mul = mul nsw i32 %17, 1
  %add = add nsw i32 0, %mul
  store i32 %add, i32* %j, align 4, !tbaa !15
  %18 = load i32*, i32** %1, align 8, !tbaa !11
  %19 = load i32, i32* %j, align 4, !tbaa !15
  %add3 = add nsw i32 1, %19
  %idxprom = sext i32 %add3 to i64
  %arrayidx = getelementptr inbounds i32, i32* %18, i64 %idxprom
  %20 = load i32, i32* %arrayidx, align 4, !tbaa !15
  %21 = load i32*, i32** %0, align 8, !tbaa !11
  %22 = load i32, i32* %j, align 4, !tbaa !15
  %idxprom4 = sext i32 %22 to i64
  %arrayidx5 = getelementptr inbounds i32, i32* %21, i64 %idxprom4
  %23 = load i32, i32* %arrayidx5, align 4, !tbaa !15
  %add6 = add nsw i32 %23, %20
  store i32 %add6, i32* %arrayidx5, align 4, !tbaa !15
  br label %omp.body.continue

omp.body.continue:                                ; preds = %omp.inner.for.body
  br label %omp.inner.for.inc

omp.inner.for.inc:                                ; preds = %omp.body.continue
  %24 = load i32, i32* %.omp.iv, align 4, !tbaa !15
  %add7 = add nsw i32 %24, 1
  store i32 %add7, i32* %.omp.iv, align 4, !tbaa !15
  br label %omp.inner.for.cond

omp.inner.for.end:                                ; preds = %omp.inner.for.cond.cleanup
  br label %omp.dispatch.inc

omp.dispatch.inc:                                 ; preds = %omp.inner.for.end
  %25 = load i32, i32* %.omp.lb, align 4, !tbaa !15
  %26 = load i32, i32* %.omp.stride, align 4, !tbaa !15
  %add8 = add nsw i32 %25, %26
  store i32 %add8, i32* %.omp.lb, align 4, !tbaa !15
  %27 = load i32, i32* %.omp.ub, align 4, !tbaa !15
  %28 = load i32, i32* %.omp.stride, align 4, !tbaa !15
  %add9 = add nsw i32 %27, %28
  store i32 %add9, i32* %.omp.ub, align 4, !tbaa !15
  br label %omp.dispatch.cond

omp.dispatch.end:                                 ; preds = %omp.dispatch.cleanup
  call void @__kmpc_for_static_fini(%struct.ident_t* @0, i32 %9)
  %29 = bitcast i32* %j to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %29) #4
  %30 = bitcast i32* %.omp.is_last to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %30) #4
  %31 = bitcast i32* %.omp.stride to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %31) #4
  %32 = bitcast i32* %.omp.ub to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %32) #4
  %33 = bitcast i32* %.omp.lb to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %33) #4
  %34 = bitcast i32* %.omp.iv to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %34) #4
  ret void
}

; Function Attrs: norecurse nounwind
define internal void @.omp_TRegion.1_wrapper(i8* %shared_vars, i8* %private_vars) #0 {
entry:
  %.addr = alloca i8*, align 8
  %.addr1 = alloca i8*, align 8
  %.zero.addr = alloca i32, align 4
  %.threadid_temp. = alloca i32, align 4
  store i32 0, i32* %.zero.addr, align 4
  store i8* %shared_vars, i8** %.addr, align 8, !tbaa !11
  store i8* %private_vars, i8** %.addr1, align 8, !tbaa !11
  %0 = call i32 @__kmpc_global_thread_num(%struct.ident_t* @1)
  store i32 %0, i32* %.threadid_temp., align 4, !tbaa !15
  %1 = bitcast i8* %private_vars to %omp.private.struct.0*
  %2 = getelementptr inbounds %omp.private.struct.0, %omp.private.struct.0* %1, i32 0, i32 0
  %3 = load i32**, i32*** %2, align 1
  %4 = getelementptr inbounds %omp.private.struct.0, %omp.private.struct.0* %1, i32 0, i32 1
  %5 = load i32**, i32*** %4, align 1
  call void @.omp_TRegion.1(i32* %.threadid_temp., i32* %.zero.addr, i32** %3, i32** %5) #4
  ret void
}

; Function Attrs: norecurse nounwind
define internal void @.omp_TRegion.2(i32* noalias %.global_tid., i32* noalias %.bound_tid., i32** dereferenceable(8) %A, i32** dereferenceable(8) %B) #0 {
entry:
  %.global_tid..addr = alloca i32*, align 8
  %.bound_tid..addr = alloca i32*, align 8
  %A.addr = alloca i32**, align 8
  %B.addr = alloca i32**, align 8
  %.omp.iv = alloca i32, align 4
  %tmp = alloca i32, align 4
  %.omp.lb = alloca i32, align 4
  %.omp.ub = alloca i32, align 4
  %.omp.stride = alloca i32, align 4
  %.omp.is_last = alloca i32, align 4
  %j = alloca i32, align 4
  store i32* %.global_tid., i32** %.global_tid..addr, align 8, !tbaa !11
  store i32* %.bound_tid., i32** %.bound_tid..addr, align 8, !tbaa !11
  store i32** %A, i32*** %A.addr, align 8, !tbaa !11
  store i32** %B, i32*** %B.addr, align 8, !tbaa !11
  %0 = load i32**, i32*** %A.addr, align 8, !tbaa !11
  %1 = load i32**, i32*** %B.addr, align 8, !tbaa !11
  %2 = bitcast i32* %.omp.iv to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %2) #4
  %3 = bitcast i32* %.omp.lb to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %3) #4
  store i32 0, i32* %.omp.lb, align 4, !tbaa !15
  %4 = bitcast i32* %.omp.ub to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %4) #4
  store i32 1023, i32* %.omp.ub, align 4, !tbaa !15
  %5 = bitcast i32* %.omp.stride to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %5) #4
  store i32 1, i32* %.omp.stride, align 4, !tbaa !15
  %6 = bitcast i32* %.omp.is_last to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %6) #4
  store i32 0, i32* %.omp.is_last, align 4, !tbaa !15
  %7 = bitcast i32* %j to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %7) #4
  %8 = load i32*, i32** %.global_tid..addr, align 8
  %9 = load i32, i32* %8, align 4, !tbaa !15
  call void @__kmpc_for_static_init_4(%struct.ident_t* @0, i32 %9, i32 33, i32* %.omp.is_last, i32* %.omp.lb, i32* %.omp.ub, i32* %.omp.stride, i32 1, i32 1)
  br label %omp.dispatch.cond

omp.dispatch.cond:                                ; preds = %omp.dispatch.inc, %entry
  %10 = load i32, i32* %.omp.ub, align 4, !tbaa !15
  %cmp = icmp sgt i32 %10, 1023
  br i1 %cmp, label %cond.true, label %cond.false

cond.true:                                        ; preds = %omp.dispatch.cond
  br label %cond.end

cond.false:                                       ; preds = %omp.dispatch.cond
  %11 = load i32, i32* %.omp.ub, align 4, !tbaa !15
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi i32 [ 1023, %cond.true ], [ %11, %cond.false ]
  store i32 %cond, i32* %.omp.ub, align 4, !tbaa !15
  %12 = load i32, i32* %.omp.lb, align 4, !tbaa !15
  store i32 %12, i32* %.omp.iv, align 4, !tbaa !15
  %13 = load i32, i32* %.omp.iv, align 4, !tbaa !15
  %14 = load i32, i32* %.omp.ub, align 4, !tbaa !15
  %cmp1 = icmp sle i32 %13, %14
  br i1 %cmp1, label %omp.dispatch.body, label %omp.dispatch.cleanup

omp.dispatch.cleanup:                             ; preds = %cond.end
  br label %omp.dispatch.end

omp.dispatch.body:                                ; preds = %cond.end
  br label %omp.inner.for.cond

omp.inner.for.cond:                               ; preds = %omp.inner.for.inc, %omp.dispatch.body
  %15 = load i32, i32* %.omp.iv, align 4, !tbaa !15
  %16 = load i32, i32* %.omp.ub, align 4, !tbaa !15
  %cmp2 = icmp sle i32 %15, %16
  br i1 %cmp2, label %omp.inner.for.body, label %omp.inner.for.cond.cleanup

omp.inner.for.cond.cleanup:                       ; preds = %omp.inner.for.cond
  br label %omp.inner.for.end

omp.inner.for.body:                               ; preds = %omp.inner.for.cond
  %17 = load i32, i32* %.omp.iv, align 4, !tbaa !15
  %mul = mul nsw i32 %17, 1
  %add = add nsw i32 0, %mul
  store i32 %add, i32* %j, align 4, !tbaa !15
  %18 = load i32*, i32** %1, align 8, !tbaa !11
  %19 = load i32, i32* %j, align 4, !tbaa !15
  %add3 = add nsw i32 2, %19
  %idxprom = sext i32 %add3 to i64
  %arrayidx = getelementptr inbounds i32, i32* %18, i64 %idxprom
  %20 = load i32, i32* %arrayidx, align 4, !tbaa !15
  %21 = load i32*, i32** %0, align 8, !tbaa !11
  %22 = load i32, i32* %j, align 4, !tbaa !15
  %idxprom4 = sext i32 %22 to i64
  %arrayidx5 = getelementptr inbounds i32, i32* %21, i64 %idxprom4
  %23 = load i32, i32* %arrayidx5, align 4, !tbaa !15
  %add6 = add nsw i32 %23, %20
  store i32 %add6, i32* %arrayidx5, align 4, !tbaa !15
  br label %omp.body.continue

omp.body.continue:                                ; preds = %omp.inner.for.body
  br label %omp.inner.for.inc

omp.inner.for.inc:                                ; preds = %omp.body.continue
  %24 = load i32, i32* %.omp.iv, align 4, !tbaa !15
  %add7 = add nsw i32 %24, 1
  store i32 %add7, i32* %.omp.iv, align 4, !tbaa !15
  br label %omp.inner.for.cond

omp.inner.for.end:                                ; preds = %omp.inner.for.cond.cleanup
  br label %omp.dispatch.inc

omp.dispatch.inc:                                 ; preds = %omp.inner.for.end
  %25 = load i32, i32* %.omp.lb, align 4, !tbaa !15
  %26 = load i32, i32* %.omp.stride, align 4, !tbaa !15
  %add8 = add nsw i32 %25, %26
  store i32 %add8, i32* %.omp.lb, align 4, !tbaa !15
  %27 = load i32, i32* %.omp.ub, align 4, !tbaa !15
  %28 = load i32, i32* %.omp.stride, align 4, !tbaa !15
  %add9 = add nsw i32 %27, %28
  store i32 %add9, i32* %.omp.ub, align 4, !tbaa !15
  br label %omp.dispatch.cond

omp.dispatch.end:                                 ; preds = %omp.dispatch.cleanup
  call void @__kmpc_for_static_fini(%struct.ident_t* @0, i32 %9)
  %29 = bitcast i32* %j to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %29) #4
  %30 = bitcast i32* %.omp.is_last to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %30) #4
  %31 = bitcast i32* %.omp.stride to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %31) #4
  %32 = bitcast i32* %.omp.ub to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %32) #4
  %33 = bitcast i32* %.omp.lb to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %33) #4
  %34 = bitcast i32* %.omp.iv to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %34) #4
  ret void
}

; Function Attrs: norecurse nounwind
define internal void @.omp_TRegion.2_wrapper(i8* %shared_vars, i8* %private_vars) #0 {
entry:
  %.addr = alloca i8*, align 8
  %.addr1 = alloca i8*, align 8
  %.zero.addr = alloca i32, align 4
  %.threadid_temp. = alloca i32, align 4
  store i32 0, i32* %.zero.addr, align 4
  store i8* %shared_vars, i8** %.addr, align 8, !tbaa !11
  store i8* %private_vars, i8** %.addr1, align 8, !tbaa !11
  %0 = call i32 @__kmpc_global_thread_num(%struct.ident_t* @1)
  store i32 %0, i32* %.threadid_temp., align 4, !tbaa !15
  %1 = bitcast i8* %private_vars to %omp.private.struct.1*
  %2 = getelementptr inbounds %omp.private.struct.1, %omp.private.struct.1* %1, i32 0, i32 0
  %3 = load i32**, i32*** %2, align 1
  %4 = getelementptr inbounds %omp.private.struct.1, %omp.private.struct.1* %1, i32 0, i32 1
  %5 = load i32**, i32*** %4, align 1
  call void @.omp_TRegion.2(i32* %.threadid_temp., i32* %.zero.addr, i32** %3, i32** %5) #4
  ret void
}

; Function Attrs: norecurse nounwind
define weak void @__omp_offloading_18_29b03e4_parallel_loop_in_function_in_loop_with_global_acc_in_tregion_l35(i32* %A, i32* %B, [512 x i32]* dereferenceable(2048) %Global) #0 {
entry:
  %A.addr = alloca i32*, align 8
  %B.addr = alloca i32*, align 8
  %Global.addr = alloca [512 x i32]*, align 8
  %i = alloca i32, align 4
  store i32* %A, i32** %A.addr, align 8, !tbaa !11
  store i32* %B, i32** %B.addr, align 8, !tbaa !11
  store [512 x i32]* %Global, [512 x i32]** %Global.addr, align 8, !tbaa !11
  %0 = load [512 x i32]*, [512 x i32]** %Global.addr, align 8, !tbaa !11
  %1 = call i8 @__kmpc_target_region_kernel_init(%struct.ident_t* @0, i1 false, i1 true, i1 true, i1 true)
  %2 = icmp eq i8 %1, 1
  br i1 %2, label %.execute, label %.exit

.execute:                                         ; preds = %entry
  %3 = bitcast i32* %i to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %3) #4
  store i32 0, i32* %i, align 4, !tbaa !15
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %.execute
  %4 = load i32, i32* %i, align 4, !tbaa !15
  %cmp = icmp slt i32 %4, 512
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  %5 = bitcast i32* %i to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %5) #4
  br label %for.end

for.body:                                         ; preds = %for.cond
  %6 = load i32*, i32** %A.addr, align 8, !tbaa !11
  %7 = load i32*, i32** %B.addr, align 8, !tbaa !11
  %8 = load i32, i32* %i, align 4, !tbaa !15
  call void @parallel_loop(i32* %6, i32* %7, i32 %8)
  %9 = load i32, i32* %i, align 4, !tbaa !15
  %idxprom = sext i32 %9 to i64
  %arrayidx = getelementptr inbounds [512 x i32], [512 x i32]* %0, i64 0, i64 %idxprom
  %10 = load i32, i32* %arrayidx, align 4, !tbaa !15
  %inc = add nsw i32 %10, 1
  store i32 %inc, i32* %arrayidx, align 4, !tbaa !15
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %11 = load i32, i32* %i, align 4, !tbaa !15
  %inc1 = add nsw i32 %11, 1
  store i32 %inc1, i32* %i, align 4, !tbaa !15
  br label %for.cond

for.end:                                          ; preds = %for.cond.cleanup
  br label %.omp.deinit

.omp.deinit:                                      ; preds = %for.end
  call void @__kmpc_target_region_kernel_deinit(%struct.ident_t* @0, i1 false, i1 true)
  br label %.exit

.exit:                                            ; preds = %.omp.deinit, %entry
  ret void
}

; Function Attrs: nounwind
define internal void @parallel_loop(i32* %A, i32* %B, i32 %i) #2 {
entry:
  %A.addr = alloca i32*, align 8
  %B.addr = alloca i32*, align 8
  %i.addr = alloca i32, align 4
  %.private.vars = alloca %omp.private.struct.2, align 8
  store i32* %A, i32** %A.addr, align 8, !tbaa !11
  store i32* %B, i32** %B.addr, align 8, !tbaa !11
  store i32 %i, i32* %i.addr, align 4, !tbaa !15
  %0 = bitcast %omp.private.struct.2* %.private.vars to i8*
  %1 = getelementptr inbounds %omp.private.struct.2, %omp.private.struct.2* %.private.vars, i32 0, i32 0
  store i32** %A.addr, i32*** %1
  %2 = getelementptr inbounds %omp.private.struct.2, %omp.private.struct.2* %.private.vars, i32 0, i32 1
  store i32** %B.addr, i32*** %2
  %3 = getelementptr inbounds %omp.private.struct.2, %omp.private.struct.2* %.private.vars, i32 0, i32 2
  store i32* %i.addr, i32** %3
  call void @__kmpc_target_region_kernel_parallel(%struct.ident_t* @0, i16 -1, i1 true, void (i8*, i8*)* @.omp_TRegion.3_wrapper, i8* undef, i16 0, i8* %0, i16 24, i1 false)
  ret void
}

; Function Attrs: norecurse nounwind
define weak void @__omp_offloading_18_29b03e4_parallel_loops_in_functions_and_extern_func_in_tregion_l43(i32* %A, i32* %B) #0 {
entry:
  %A.addr = alloca i32*, align 8
  %B.addr = alloca i32*, align 8
  store i32* %A, i32** %A.addr, align 8, !tbaa !11
  store i32* %B, i32** %B.addr, align 8, !tbaa !11
  %0 = call i8 @__kmpc_target_region_kernel_init(%struct.ident_t* @0, i1 false, i1 true, i1 true, i1 true)
  %1 = icmp eq i8 %0, 1
  br i1 %1, label %.execute, label %.exit

.execute:                                         ; preds = %entry
  %2 = load i32*, i32** %A.addr, align 8, !tbaa !11
  %3 = load i32*, i32** %B.addr, align 8, !tbaa !11
  call void @parallel_loop(i32* %2, i32* %3, i32 1)
  %4 = load i32*, i32** %A.addr, align 8, !tbaa !11
  %5 = load i32*, i32** %B.addr, align 8, !tbaa !11
  call void @parallel_loop(i32* %4, i32* %5, i32 2)
  call void bitcast (void (...)* @extern_func to void ()*)()
  %6 = load i32*, i32** %A.addr, align 8, !tbaa !11
  %7 = load i32*, i32** %B.addr, align 8, !tbaa !11
  call void @parallel_loop(i32* %6, i32* %7, i32 3)
  br label %.omp.deinit

.omp.deinit:                                      ; preds = %.execute
  call void @__kmpc_target_region_kernel_deinit(%struct.ident_t* @0, i1 false, i1 true)
  br label %.exit

.exit:                                            ; preds = %.omp.deinit, %entry
  ret void
}

declare void @extern_func(...) #3

; Function Attrs: norecurse nounwind
define internal void @.omp_TRegion.3(i32* noalias %.global_tid., i32* noalias %.bound_tid., i32** dereferenceable(8) %A, i32** dereferenceable(8) %B, i32* dereferenceable(4) %i) #0 {
entry:
  %.global_tid..addr = alloca i32*, align 8
  %.bound_tid..addr = alloca i32*, align 8
  %A.addr = alloca i32**, align 8
  %B.addr = alloca i32**, align 8
  %i.addr = alloca i32*, align 8
  %.omp.iv = alloca i32, align 4
  %tmp = alloca i32, align 4
  %.omp.lb = alloca i32, align 4
  %.omp.ub = alloca i32, align 4
  %.omp.stride = alloca i32, align 4
  %.omp.is_last = alloca i32, align 4
  %j = alloca i32, align 4
  store i32* %.global_tid., i32** %.global_tid..addr, align 8, !tbaa !11
  store i32* %.bound_tid., i32** %.bound_tid..addr, align 8, !tbaa !11
  store i32** %A, i32*** %A.addr, align 8, !tbaa !11
  store i32** %B, i32*** %B.addr, align 8, !tbaa !11
  store i32* %i, i32** %i.addr, align 8, !tbaa !11
  %0 = load i32**, i32*** %A.addr, align 8, !tbaa !11
  %1 = load i32**, i32*** %B.addr, align 8, !tbaa !11
  %2 = load i32*, i32** %i.addr, align 8, !tbaa !11
  %3 = bitcast i32* %.omp.iv to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %3) #4
  %4 = bitcast i32* %.omp.lb to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %4) #4
  store i32 0, i32* %.omp.lb, align 4, !tbaa !15
  %5 = bitcast i32* %.omp.ub to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %5) #4
  store i32 1023, i32* %.omp.ub, align 4, !tbaa !15
  %6 = bitcast i32* %.omp.stride to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %6) #4
  store i32 1, i32* %.omp.stride, align 4, !tbaa !15
  %7 = bitcast i32* %.omp.is_last to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %7) #4
  store i32 0, i32* %.omp.is_last, align 4, !tbaa !15
  %8 = bitcast i32* %j to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %8) #4
  %9 = load i32*, i32** %.global_tid..addr, align 8
  %10 = load i32, i32* %9, align 4, !tbaa !15
  call void @__kmpc_for_static_init_4(%struct.ident_t* @0, i32 %10, i32 33, i32* %.omp.is_last, i32* %.omp.lb, i32* %.omp.ub, i32* %.omp.stride, i32 1, i32 1)
  br label %omp.dispatch.cond

omp.dispatch.cond:                                ; preds = %omp.dispatch.inc, %entry
  %11 = load i32, i32* %.omp.ub, align 4, !tbaa !15
  %cmp = icmp sgt i32 %11, 1023
  br i1 %cmp, label %cond.true, label %cond.false

cond.true:                                        ; preds = %omp.dispatch.cond
  br label %cond.end

cond.false:                                       ; preds = %omp.dispatch.cond
  %12 = load i32, i32* %.omp.ub, align 4, !tbaa !15
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi i32 [ 1023, %cond.true ], [ %12, %cond.false ]
  store i32 %cond, i32* %.omp.ub, align 4, !tbaa !15
  %13 = load i32, i32* %.omp.lb, align 4, !tbaa !15
  store i32 %13, i32* %.omp.iv, align 4, !tbaa !15
  %14 = load i32, i32* %.omp.iv, align 4, !tbaa !15
  %15 = load i32, i32* %.omp.ub, align 4, !tbaa !15
  %cmp1 = icmp sle i32 %14, %15
  br i1 %cmp1, label %omp.dispatch.body, label %omp.dispatch.cleanup

omp.dispatch.cleanup:                             ; preds = %cond.end
  br label %omp.dispatch.end

omp.dispatch.body:                                ; preds = %cond.end
  br label %omp.inner.for.cond

omp.inner.for.cond:                               ; preds = %omp.inner.for.inc, %omp.dispatch.body
  %16 = load i32, i32* %.omp.iv, align 4, !tbaa !15
  %17 = load i32, i32* %.omp.ub, align 4, !tbaa !15
  %cmp2 = icmp sle i32 %16, %17
  br i1 %cmp2, label %omp.inner.for.body, label %omp.inner.for.cond.cleanup

omp.inner.for.cond.cleanup:                       ; preds = %omp.inner.for.cond
  br label %omp.inner.for.end

omp.inner.for.body:                               ; preds = %omp.inner.for.cond
  %18 = load i32, i32* %.omp.iv, align 4, !tbaa !15
  %mul = mul nsw i32 %18, 1
  %add = add nsw i32 0, %mul
  store i32 %add, i32* %j, align 4, !tbaa !15
  %19 = load i32*, i32** %1, align 8, !tbaa !11
  %20 = load i32, i32* %2, align 4, !tbaa !15
  %21 = load i32, i32* %j, align 4, !tbaa !15
  %add3 = add nsw i32 %20, %21
  %idxprom = sext i32 %add3 to i64
  %arrayidx = getelementptr inbounds i32, i32* %19, i64 %idxprom
  %22 = load i32, i32* %arrayidx, align 4, !tbaa !15
  %23 = load i32*, i32** %0, align 8, !tbaa !11
  %24 = load i32, i32* %j, align 4, !tbaa !15
  %idxprom4 = sext i32 %24 to i64
  %arrayidx5 = getelementptr inbounds i32, i32* %23, i64 %idxprom4
  %25 = load i32, i32* %arrayidx5, align 4, !tbaa !15
  %add6 = add nsw i32 %25, %22
  store i32 %add6, i32* %arrayidx5, align 4, !tbaa !15
  br label %omp.body.continue

omp.body.continue:                                ; preds = %omp.inner.for.body
  br label %omp.inner.for.inc

omp.inner.for.inc:                                ; preds = %omp.body.continue
  %26 = load i32, i32* %.omp.iv, align 4, !tbaa !15
  %add7 = add nsw i32 %26, 1
  store i32 %add7, i32* %.omp.iv, align 4, !tbaa !15
  br label %omp.inner.for.cond

omp.inner.for.end:                                ; preds = %omp.inner.for.cond.cleanup
  br label %omp.dispatch.inc

omp.dispatch.inc:                                 ; preds = %omp.inner.for.end
  %27 = load i32, i32* %.omp.lb, align 4, !tbaa !15
  %28 = load i32, i32* %.omp.stride, align 4, !tbaa !15
  %add8 = add nsw i32 %27, %28
  store i32 %add8, i32* %.omp.lb, align 4, !tbaa !15
  %29 = load i32, i32* %.omp.ub, align 4, !tbaa !15
  %30 = load i32, i32* %.omp.stride, align 4, !tbaa !15
  %add9 = add nsw i32 %29, %30
  store i32 %add9, i32* %.omp.ub, align 4, !tbaa !15
  br label %omp.dispatch.cond

omp.dispatch.end:                                 ; preds = %omp.dispatch.cleanup
  call void @__kmpc_for_static_fini(%struct.ident_t* @0, i32 %10)
  %31 = bitcast i32* %j to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %31) #4
  %32 = bitcast i32* %.omp.is_last to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %32) #4
  %33 = bitcast i32* %.omp.stride to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %33) #4
  %34 = bitcast i32* %.omp.ub to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %34) #4
  %35 = bitcast i32* %.omp.lb to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %35) #4
  %36 = bitcast i32* %.omp.iv to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %36) #4
  ret void
}

; Function Attrs: norecurse nounwind
define internal void @.omp_TRegion.3_wrapper(i8* %shared_vars, i8* %private_vars) #0 {
entry:
  %.addr = alloca i8*, align 8
  %.addr1 = alloca i8*, align 8
  %.zero.addr = alloca i32, align 4
  %.threadid_temp. = alloca i32, align 4
  store i32 0, i32* %.zero.addr, align 4
  store i8* %shared_vars, i8** %.addr, align 8, !tbaa !11
  store i8* %private_vars, i8** %.addr1, align 8, !tbaa !11
  %0 = call i32 @__kmpc_global_thread_num(%struct.ident_t* @1)
  store i32 %0, i32* %.threadid_temp., align 4, !tbaa !15
  %1 = bitcast i8* %private_vars to %omp.private.struct.2*
  %2 = getelementptr inbounds %omp.private.struct.2, %omp.private.struct.2* %1, i32 0, i32 0
  %3 = load i32**, i32*** %2, align 1
  %4 = getelementptr inbounds %omp.private.struct.2, %omp.private.struct.2* %1, i32 0, i32 1
  %5 = load i32**, i32*** %4, align 1
  %6 = getelementptr inbounds %omp.private.struct.2, %omp.private.struct.2* %1, i32 0, i32 2
  %7 = load i32*, i32** %6, align 1
  call void @.omp_TRegion.3(i32* %.threadid_temp., i32* %.zero.addr, i32** %3, i32** %5, i32* %7) #4
  ret void
}

attributes #0 = { norecurse nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="sm_35" "target-features"="+ptx32,+sm_35" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="sm_35" "target-features"="+ptx32,+sm_35" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="sm_35" "target-features"="+ptx32,+sm_35" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind }

!omp_offload.info = !{!0, !1, !2, !3}
!nvvm.annotations = !{!4, !5, !6, !7}
!llvm.module.flags = !{!8, !9}
!llvm.ident = !{!10}

!0 = !{i32 0, i32 24, i32 43713508, !"parallel_loops_and_accesses_in_tregion", i32 9, i32 1}
!1 = !{i32 0, i32 24, i32 43713508, !"loop_in_loop_in_tregion", i32 2, i32 0}
!2 = !{i32 0, i32 24, i32 43713508, !"parallel_loops_in_functions_and_extern_func_in_tregion", i32 43, i32 3}
!3 = !{i32 0, i32 24, i32 43713508, !"parallel_loop_in_function_in_loop_with_global_acc_in_tregion", i32 35, i32 2}
!4 = !{void (i32*, i32*)* @__omp_offloading_18_29b03e4_loop_in_loop_in_tregion_l2, !"kernel", i32 1}
!5 = !{void (i32*, i32*)* @__omp_offloading_18_29b03e4_parallel_loops_and_accesses_in_tregion_l9, !"kernel", i32 1}
!6 = !{void (i32*, i32*, [512 x i32]*)* @__omp_offloading_18_29b03e4_parallel_loop_in_function_in_loop_with_global_acc_in_tregion_l35, !"kernel", i32 1}
!7 = !{void (i32*, i32*)* @__omp_offloading_18_29b03e4_parallel_loops_in_functions_and_extern_func_in_tregion_l43, !"kernel", i32 1}
!8 = !{i32 1, !"wchar_size", i32 4}
!9 = !{i32 7, !"PIC Level", i32 2}
!10 = !{!"clang version 9.0.0 "}
!11 = !{!12, !12, i64 0}
!12 = !{!"any pointer", !13, i64 0}
!13 = !{!"omnipotent char", !14, i64 0}
!14 = !{!"Simple C/C++ TBAA"}
!15 = !{!16, !16, i64 0}
!16 = !{!"int", !13, i64 0}
!17 = !{!18}
!18 = !{i64 2, i64 3, i64 5, i1 false}

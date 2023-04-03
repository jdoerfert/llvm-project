; ModuleID = 'test-openmp-nvptx64-nvidia-cuda-sm_75.bc'
source_filename = "/home/shiltian/Documents/vscode/llvm-project/auto-host-rpc/test.c"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%struct.ddd = type { i32, i32, float }

@__omp_rtl_debug_kind = weak_odr hidden local_unnamed_addr constant i32 0
@__omp_rtl_assume_teams_oversubscription = weak_odr hidden local_unnamed_addr constant i32 0
@__omp_rtl_assume_threads_oversubscription = weak_odr hidden local_unnamed_addr constant i32 0
@__omp_rtl_assume_no_thread_state = weak_odr hidden local_unnamed_addr constant i32 0
@__omp_rtl_assume_no_nested_parallelism = weak_odr hidden local_unnamed_addr constant i32 0
@.str = private unnamed_addr constant [9 x i8] c"main.cpp\00", align 1
@.str1 = private unnamed_addr constant [2 x i8] c"r\00", align 1
@.str2 = private unnamed_addr constant [3 x i8] c"%d\00", align 1
@.str3 = private unnamed_addr constant [7 x i8] c"%f%d%s\00", align 1
@.str4 = private unnamed_addr constant [6 x i8] c"hello\00", align 1

; Function Attrs: convergent nounwind
define hidden void @foo() local_unnamed_addr #0 {
entry:
  %d = tail call align 16 dereferenceable_or_null(12) ptr @__kmpc_alloc_shared(i64 12) #4
  %call = tail call noalias ptr @fopen(ptr noundef nonnull @.str, ptr noundef nonnull @.str1) #5
  %call1 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %call, ptr noundef nonnull @.str2, i32 noundef 6) #5
  %call2 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %call, ptr noundef nonnull @.str3, double noundef 6.000000e+00, i32 noundef 1, ptr noundef nonnull @.str4) #5
  %a = getelementptr inbounds %struct.ddd, ptr %d, i64 0, i32 1
  %call3 = tail call i32 (ptr, ptr, ...) @fscanf(ptr noundef %call, ptr noundef nonnull @.str2, ptr noundef nonnull %a) #5
  tail call void @__kmpc_free_shared(ptr %d, i64 12)
  ret void
}

; Function Attrs: nofree nosync nounwind allocsize(0)
declare ptr @__kmpc_alloc_shared(i64) local_unnamed_addr #1

; Function Attrs: convergent
declare noalias ptr @fopen(ptr noundef, ptr noundef) local_unnamed_addr #2

; Function Attrs: convergent
declare i32 @fprintf(ptr noundef, ptr noundef, ...) local_unnamed_addr #2

; Function Attrs: convergent
declare i32 @fscanf(ptr noundef, ptr noundef, ...) local_unnamed_addr #2

; Function Attrs: nosync nounwind
declare void @__kmpc_free_shared(ptr allocptr nocapture, i64) local_unnamed_addr #3

attributes #0 = { convergent nounwind "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_75" "target-features"="+ptx77,+sm_75" }
attributes #1 = { nofree nosync nounwind allocsize(0) }
attributes #2 = { convergent "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_75" "target-features"="+ptx77,+sm_75" }
attributes #3 = { nosync nounwind }
attributes #4 = { nounwind }
attributes #5 = { convergent nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5}
!llvm.ident = !{!6, !7}
!nvvm.annotations = !{}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 11, i32 7]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"openmp", i32 50}
!3 = !{i32 7, !"openmp-device", i32 50}
!4 = !{i32 8, !"PIC Level", i32 2}
!5 = !{i32 7, !"frame-pointer", i32 2}
!6 = !{!"clang version 16.0.0"}
!7 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}

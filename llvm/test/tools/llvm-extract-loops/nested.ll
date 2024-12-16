; RUN: mkdir -p %t
; RUN: rm %t/extracted.* || true
; RUN: llvm-extract-loops -S %s --output-prefix %t/extracted. --output-suffix .ll --pretty-print-json
; RUN: cat %t/extracted.*.ll | FileCheck %s
; RUN: cat %t/extracted.0.ll.json | FileCheck %s --check-prefix=LOOP_0
; RUN: cat %t/extracted.1.ll.json | FileCheck %s --check-prefix=LOOP_1

; CHECK: define{{.*}}@__llvm_extracted_loop
; CHECK: define{{.*}}@__llvm_extracted_loop

; LOOP_0-DAG:  "loop_depth": 1,
; LOOP_0-DAG:  "loop_id": 0,
; LOOP_0-DAG:  "loop_trip_count": "dynamic",
; LOOP_0-DAG:  "num_inner_loops": 1,
; LOOP_0-DAG:  "parent_function": "foo",
; LOOP_0-DAG:  "parent_loop_id": -1

; LOOP_1-DAG:  "loop_depth": 2,
; LOOP_1-DAG:  "loop_id": 1,
; LOOP_1-DAG:  "loop_trip_count": "dynamic",
; LOOP_1-DAG:  "num_inner_loops": 0,
; LOOP_1-DAG:  "parent_function": "foo",
; LOOP_1-DAG:  "parent_loop_id": 0

define i32 @foo(ptr %array, i32 %length, i32 %n, i32 %l) {
entry:
  %tmp5 = icmp sle i32 %n, 0
  br i1 %tmp5, label %exit, label %outer.loop.preheader

outer.loop.preheader:
  br label %outer.loop

outer.loop:
  %outer.loop.acc = phi i32 [ %outer.loop.acc.next, %outer.loop.inc ], [ 0, %outer.loop.preheader ]
  %i = phi i32 [ %i.next, %outer.loop.inc ], [ 0, %outer.loop.preheader ]
  %tmp6 = icmp sle i32 %l, 0
  br i1 %tmp6, label %outer.loop.inc, label %inner.loop.preheader

inner.loop.preheader:
  br label %inner.loop

inner.loop:
  %inner.loop.acc = phi i32 [ %inner.loop.acc.next, %inner.loop ], [ %outer.loop.acc, %inner.loop.preheader ]
  %j = phi i32 [ %j.next, %inner.loop ], [ 0, %inner.loop.preheader ]

  %within.bounds = icmp ult i32 %j, %length
  call void (i1, ...) @llvm.experimental.guard(i1 %within.bounds, i32 9) [ "deopt"() ]

  %j.i64 = zext i32 %j to i64
  %array.j.ptr = getelementptr inbounds i32, ptr %array, i64 %j.i64
  %array.j = load i32, ptr %array.j.ptr, align 4
  %inner.loop.acc.next = add i32 %inner.loop.acc, %array.j

  %j.next = add nsw i32 %j, 1
  %inner.continue = icmp slt i32 %j.next, %l
  br i1 %inner.continue, label %inner.loop, label %outer.loop.inc

outer.loop.inc:
  %outer.loop.acc.next = phi i32 [ %inner.loop.acc.next, %inner.loop ], [ %outer.loop.acc, %outer.loop ]
  %i.next = add nsw i32 %i, 1
  %outer.continue = icmp slt i32 %i.next, %n
  br i1 %outer.continue, label %outer.loop, label %exit

exit:
  %result = phi i32 [ 0, %entry ], [ %outer.loop.acc.next, %outer.loop.inc ]
  ret i32 %result
}

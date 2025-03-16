; RUN: opt -S --input-gen-mode=generate --passes=input-gen-instrument-entries,inline,input-gen-instrument-memory %s | FileCheck %s

define i32 @load_from_ptr(ptr %a) #0 {
  %v = load i32, ptr %a
  ret i32 %v
}

attributes #0 = { inputgen_entry }

; CHECK-LABEL: define void @__ig_entry(
; CHECK-SAME: i32 [[TMP0:%.*]], ptr [[TMP1:%.*]]) {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    switch i32 [[TMP0]], label %[[RETURN:.*]] [
; CHECK-NEXT:      i32 0, label %[[DISPATCH:.*]]
; CHECK-NEXT:    ]
; CHECK:       [[RETURN]]:
; CHECK-NEXT:    ret void
; CHECK:       [[DISPATCH]]:
; CHECK-NEXT:    musttail call void @__ig_ig_entry_func.load_from_ptr.wrapper(i32 [[TMP0]], ptr [[TMP1]])
; CHECK-NEXT:    ret void
;
;
; CHECK-LABEL: define internal void @__ig_ig_entry_func.load_from_ptr.wrapper(
; CHECK-SAME: i32 [[TMP0:%.*]], ptr [[TMP1:%.*]])
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[TMP2:%.*]] = getelementptr ptr, ptr [[TMP1]], i32 8
; CHECK-NEXT:    [[TMP3:%.*]] = call ptr @__ig_post_base_pointer_info(ptr [[TMP1]], i32 0)
; CHECK-NEXT:    [[TMP4:%.*]] = call ptr @__ig_pre_store(ptr [[TMP2]], ptr [[TMP3]], ptr null, i64 4, i64 4, i32 12)
; CHECK-NEXT:    [[TMP5:%.*]] = call ptr @__ig_pre_load(ptr [[TMP1]], ptr [[TMP3]], ptr null, i64 8, i64 8, i32 14)
; CHECK-NEXT:    [[A:%.*]] = load ptr, ptr [[TMP5]], align 8
; CHECK-NEXT:    [[TMP6:%.*]] = call ptr @__ig_post_base_pointer_info(ptr [[A]], i32 2)
; CHECK-NEXT:    [[TMP7:%.*]] = call ptr @__ig_pre_load(ptr [[A]], ptr [[TMP6]], ptr null, i64 4, i64 4, i32 12)
; CHECK-NEXT:    [[V_I:%.*]] = load i32, ptr [[TMP7]], align 4
; CHECK-NEXT:    store i32 [[V_I]], ptr [[TMP4]], align 4
; CHECK-NEXT:    ret void

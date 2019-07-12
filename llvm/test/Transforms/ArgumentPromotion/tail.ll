; RUN: opt %s -argpromotion -S -o - | FileCheck %s
; RUN: opt %s -passes=argpromotion -S -o - | FileCheck %s
; PR14710, and related problems (baz, biz, buz) where 'tail' needs to be
; removed if we introduce allocas.

; FIXME: If the function is norecurse 'tail' removal should not be necessary.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

%pair = type { i32, i32 }

declare i8* @foo(%pair*)

define internal void @bar(%pair* byval %Data) {
; CHECK: define internal void @bar(i32 %{{.*}}, i32 %{{.*}})
; CHECK: %Data = alloca %pair
; CHECK-NOT: tail
; CHECK: call i8* @foo(%pair* %Data)
  tail call i8* @foo(%pair* %Data)
  ret void
}

define internal void @baz(%pair* byval %Data) {
; CHECK: define internal void @baz(i32 %{{.*}}, i32 %{{.*}})
; CHECK: %Data = alloca %pair
; CHECK: %Data2 = getelementptr %pair, %pair* %Data
; CHECK-NOT: tail
; CHECK: call i8* @foo(%pair* %Data2)
  %Data2 = getelementptr %pair, %pair* %Data
  tail call i8* @foo(%pair* %Data2)
  ret void
}

@a = global %pair* null, align 8
declare void @unknown(%pair*)

define internal void @biz(%pair* byval %Data) {
; CHECK: define internal void @biz(i32 %{{.*}}, i32 %{{.*}})
; CHECK: %Data = alloca %pair
; CHECK: %Data2 = load %pair*, %pair** @a
; CHECK-NOT: tail
; CHECK: call i8* @foo(%pair* %Data2)
  call void @unknown(%pair* %Data)
  %Data2 = load %pair*, %pair** @a
  tail call i8* @foo(%pair* %Data2)
  ret void
}

define internal void @buz(%pair* byval %Data) {
; CHECK: define internal void @buz(i32 %{{.*}}, i32 %{{.*}})
; CHECK: %Data = alloca %pair
; CHECK: call i8* @foo(%pair* %Data)
; CHECK-NOT: tail
; CHECK: call i8* @foo(%pair* %Data2)
  %fr = call i8* @foo(%pair* %Data)
  %Data2 = bitcast i8* %fr to %pair*
  tail call i8* @foo(%pair* %Data2)
  ret void
}

define void @zed(%pair* byval %Data) {
  call void @bar(%pair* byval %Data)
  call void @baz(%pair* byval %Data)
  call void @biz(%pair* byval %Data)
  call void @buz(%pair* byval %Data)
  ret void
}

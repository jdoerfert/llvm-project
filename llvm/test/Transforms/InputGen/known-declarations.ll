; RUN: opt -S --input-gen-mode=generate --passes=input-gen-instrument-entries,input-gen-instrument-memory %s | FileCheck %s

; CHECK: declare i32 @__ig_known_memcmp
; CHECK: declare i32 @__ig_known_strcmp

declare i32 @memcmp(ptr noundef, ptr noundef, i64 noundef)
declare i32 @strcmp(ptr noundef, ptr noundef)

define void @foo(ptr %a, ptr %b, i64 %n) #0 {
  %c = call i32 (ptr, ptr, i64) @memcmp(ptr %a, ptr %b, i64 %n)
  %d = call i32 (ptr, ptr) @strcmp(ptr %a, ptr %b)
  ret void
}

attributes #0 = { inputgen_entry }

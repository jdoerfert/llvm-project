; RUN: opt -S < %s | FileCheck %s

declare void @foo()

define void @check_lines_1() {
  ret void
}

; UTC_ARGS: --turn=off

define void @no_check_lines() {
  ret void
}

; UTC_ARGS: --turn=on

define void @check_lines_2() {
  ret void
}

define void @scrub() {
  call void @foo() readnone
  ret void
}

define i32 @signature(i32 %arg) {
  ret i32 %arg
}

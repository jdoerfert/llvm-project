; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

declare i8* @foo()

define void @f1() {
entry:
  call i8* @foo(), !dereferenceable_globally !{i64 2}
  ret void
}
; CHECK: dereferenceable_globally, dereferenceable_or_null_globally apply only to load instructions, use attributes for calls or invokes
; CHECK-NEXT: call i8* @foo()

define void @f2() {
entry:
  call i8* @foo(), !dereferenceable_or_null_globally !{i64 2}
  ret void
}
; CHECK: dereferenceable_globally, dereferenceable_or_null_globally apply only to load instructions, use attributes for calls or invokes
; CHECK-NEXT: call i8* @foo()

define i8 @f3(i8* %x) {
entry:
  %y = load i8, i8* %x, !dereferenceable_globally !{i64 2}
  ret i8 %y
}
; CHECK: dereferenceable_globally, dereferenceable_or_null_globally apply only to pointer types
; CHECK-NEXT: load i8, i8* %x

define i8 @f4(i8* %x) {
entry:
  %y = load i8, i8* %x, !dereferenceable_or_null_globally !{i64 2}
  ret i8 %y
}
; CHECK: dereferenceable_globally, dereferenceable_or_null_globally apply only to pointer types
; CHECK-NEXT: load i8, i8* %x

define i8* @f5(i8** %x) {
entry:
  %y = load i8*, i8** %x, !dereferenceable_globally !{}
  ret i8* %y
}
; CHECK: dereferenceable_globally, dereferenceable_or_null_globally take one operand
; CHECK-NEXT: load i8*, i8** %x


define i8* @f6(i8** %x) {
entry:
  %y = load i8*, i8** %x, !dereferenceable_or_null_globally !{}
  ret i8* %y
}
; CHECK: dereferenceable_globally, dereferenceable_or_null_globally take one operand
; CHECK-NEXT: load i8*, i8** %x

define i8* @f7(i8** %x) {
entry:
  %y = load i8*, i8** %x, !dereferenceable_globally !{!"str"}
  ret i8* %y
}
; CHECK: dereferenceable_globally, dereferenceable_or_null_globally metadata value must be an i64!
; CHECK-NEXT: load i8*, i8** %x


define i8* @f8(i8** %x) {
entry:
  %y = load i8*, i8** %x, !dereferenceable_or_null_globally !{!"str"}
  ret i8* %y
}
; CHECK: dereferenceable_globally, dereferenceable_or_null_globally metadata value must be an i64!
; CHECK-NEXT: load i8*, i8** %x

define i8* @f9(i8** %x) {
entry:
  %y = load i8*, i8** %x, !dereferenceable_globally !{i32 2}
  ret i8* %y
}
; CHECK: dereferenceable_globally, dereferenceable_or_null_globally metadata value must be an i64!
; CHECK-NEXT: load i8*, i8** %x


define i8* @f10(i8** %x) {
entry:
  %y = load i8*, i8** %x, !dereferenceable_or_null_globally !{i32 2}
  ret i8* %y
}
; CHECK: dereferenceable_globally, dereferenceable_or_null_globally metadata value must be an i64!
; CHECK-NEXT: load i8*, i8** %x

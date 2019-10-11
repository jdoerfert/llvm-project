; RUN: opt -S < %s | FileCheck %s --check-prefixes=ALL,FIRST
; RUN: opt -S -globalopt < %s | FileCheck %s --check-prefixes=ALL,SECOND
;
; Make sure we use FIRST to check for @f as ALL does not work.

define internal void @f() {
  ret void
}

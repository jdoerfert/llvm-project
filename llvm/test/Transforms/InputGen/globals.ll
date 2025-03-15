; RUN: opt -S --input-gen-mode=generate --passes=input-gen-instrument-entries,input-gen-instrument-memory %s | FileCheck %s

; All non special globals should become private
; CHECK: @external_global = private global i32 0
; CHECK: @internal_global = private global i32 0

@external_global = external global i32
@internal_global = internal global i32 0

; We should change external constants to be non-constant
; CHECK: @external_constant_global = private global i32 0
@external_constant_global = external constant i32

; We need to get rid of {c,d}tors
; CHECK-NOT: @llvm.global_ctors{{.*}}ctorfunc
; CHECK-NOT: @llvm.global_dtors{{.*}}ctorfunc
@llvm.global_ctors = appending addrspace(1) global [1 x { i32, ptr, ptr  }] [{ i32, ptr, ptr  } { i32 1, ptr @ctorfunc, ptr null  }]
@llvm.global_dtors = appending addrspace(1) global [1 x { i32, ptr, ptr  }] [{ i32, ptr, ptr  } { i32 1, ptr @ctorfunc, ptr null  }]
define internal void @ctorfunc() {
      ret void
}

define i32 @foo() #0 {
  %v = load i32, ptr @external_global
  store i32 %v, ptr @internal_global
  %r = load i32, ptr @internal_global
  ret i32 %r
}

define i32 @bar() #0 {
  %v = load i32, ptr @external_constant_global
  ret i32 %v
}

attributes #0 = { inputgen_entry }

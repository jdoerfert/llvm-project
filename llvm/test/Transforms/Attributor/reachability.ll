
define dso_local void @non_recursive_asm_fn() #0 {
entry:
  call void asm sideeffect "barrier.sync $0;", "r,~{memory},~{dirflag},~{fpsr},~{flags}"(i32 1)
  ret void
}

define dso_local void @non_recursive_asm_cs() {
entry:
  call void asm sideeffect "barrier.sync $0;", "r,~{memory},~{dirflag},~{fpsr},~{flags}"(i32 1) #0
  ret void
}

define dso_local void @recursive_asm() {
entry:
  call void asm sideeffect "barrier.sync $0;", "r,~{memory},~{dirflag},~{fpsr},~{flags}"(i32 1)
  ret void
}

attributes #0 = { "llvm.assume"="ompx_no_call_asm" }

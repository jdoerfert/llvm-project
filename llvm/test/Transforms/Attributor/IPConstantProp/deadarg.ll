; RUN: opt -passes=attributor -aa-pipeline='basic-aa' -attributor-disable=false -attributor-max-iterations-verify -attributor-max-iterations=1 -disable-output < %s
define internal void @foo(i32 %X) {
        call void @foo( i32 %X )
        ret void
}


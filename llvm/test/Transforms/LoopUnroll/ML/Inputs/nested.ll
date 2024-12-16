declare i32 @foo(ptr, i32, i32, i32)

declare void @llvm.experimental.guard(i1, ...)

define hidden void @__llvm_extracted_loop(i32 %l, i32 %length, ptr %array, i32 %n, ptr %outer.loop.acc.next.out) {
newFuncRoot:
  br label %outer.loop

outer.loop:                                       ; preds = %newFuncRoot, %outer.loop.inc
  %outer.loop.acc = phi i32 [ %outer.loop.acc.next, %outer.loop.inc ], [ 0, %newFuncRoot ]
  %i = phi i32 [ %i.next, %outer.loop.inc ], [ 0, %newFuncRoot ]
  %tmp6 = icmp slt i32 %l, 1
  br i1 %tmp6, label %outer.loop.inc, label %inner.loop.preheader

inner.loop.preheader:                             ; preds = %outer.loop
  br label %inner.loop

inner.loop:                                       ; preds = %inner.loop, %inner.loop.preheader
  %inner.loop.acc = phi i32 [ %inner.loop.acc.next, %inner.loop ], [ %outer.loop.acc, %inner.loop.preheader ]
  %j = phi i32 [ %j.next, %inner.loop ], [ 0, %inner.loop.preheader ]
  %within.bounds = icmp ult i32 %j, %length
  call void (i1, ...) @llvm.experimental.guard(i1 %within.bounds, i32 9) [ "deopt"() ]
  %j.i64 = zext nneg i32 %j to i64
  %array.j.ptr = getelementptr inbounds nuw i32, ptr %array, i64 %j.i64
  %array.j = load i32, ptr %array.j.ptr, align 4
  %inner.loop.acc.next = add i32 %inner.loop.acc, %array.j
  %j.next = add nuw nsw i32 %j, 1
  %inner.continue = icmp slt i32 %j.next, %l
  br i1 %inner.continue, label %inner.loop, label %outer.loop.inc.loopexit

outer.loop.inc.loopexit:                          ; preds = %inner.loop
  br label %outer.loop.inc

outer.loop.inc:                                   ; preds = %outer.loop.inc.loopexit, %outer.loop
  %outer.loop.acc.next = phi i32 [ %outer.loop.acc, %outer.loop ], [ %inner.loop.acc.next, %outer.loop.inc.loopexit ]
  store i32 %outer.loop.acc.next, ptr %outer.loop.acc.next.out, align 4
  %i.next = add nuw nsw i32 %i, 1
  %outer.continue = icmp slt i32 %i.next, %n
  br i1 %outer.continue, label %outer.loop, label %exit.loopexit.exitStub

exit.loopexit.exitStub:                           ; preds = %outer.loop.inc
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none)) #0

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr captures(none)) #0

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }

; RUN: opt -S -gvn-hoist -argpromotion %s | FileCheck %s
;
; We should promote %x here as it is only loaded and passed into recursive calls.
; This should work after -gvn-hoist (to move the loads into the entry block) and after
; we derive dereferenceable based on the loads in all paths. The latter is
; under construction right now (in the Attributor framework).
;
; PR887

; FIXME: This should be:  define internal i32 @foo(i32 %{{.*}}, i32 %n, i32 %m)
; CHECK: define internal i32 @foo(i32* %x, i32 %n, i32 %m)
define internal i32 @foo(i32* %x, i32 %n, i32 %m) {
entry:
  %tmp = icmp ne i32 %n, 0
  br i1 %tmp, label %cond_true, label %cond_false

cond_true:                                        ; preds = %entry
  %tmp2 = load i32, i32* %x
  br label %return

cond_false:                                       ; preds = %entry
  %tmp5 = load i32, i32* %x
  %tmp7 = sub i32 %n, 1
  %tmp9 = call i32 @foo(i32* %x, i32 %tmp7, i32 %tmp5)
  %tmp11 = sub i32 %n, 2
  %tmp14 = call i32 @foo(i32* %x, i32 %tmp11, i32 %m)
  %tmp15 = add i32 %tmp9, %tmp14
  br label %return

return:                                           ; preds = %cond_next, %cond_false, %cond_true
  %retval.0 = phi i32 [ %tmp2, %cond_true ], [ %tmp15, %cond_false ]
  ret i32 %retval.0
}

define i32 @bar(i32* %x, i32 %n, i32 %m) {
entry:

; FIXME: This should be:
;   %[[XVal:[a-zA-Z._0-9]*]] = load i32, i32* %x
;   %tmp3 = call i32 @foo(i32 %[[XVal]], i32 %n, i32 %m)
; CHECK:  %tmp3 = call i32 @foo(i32* %x, i32 %n, i32 %m)
  %tmp3 = call i32 @foo(i32* %x, i32 %n, i32 %m)
  br label %return

return:                                           ; preds = %entry
  ret i32 %tmp3
}

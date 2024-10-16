; NOTE: Assertions have been autogenerated by utils/update_test_checks.py UTC_ARGS: --function-signature --scrub-attributes --check-attributes
; REQUIRES: asserts
; RUN: opt -S -passes=attributor --attributor-seed-allow-list asd < %s | FileCheck %s --check-prefixes=CHECK_DISABLED
; RUN: opt -S -passes=attributor --attributor-seed-allow-list AAValueSimplify < %s | FileCheck %s --check-prefixes=CHECK_ENABLED
; RUN: opt -S -passes=attributor --attributor-seed-allow-list=AAIsDead < %s | FileCheck %s --check-prefixes=CHECK_ISDEAD

; RUN: opt -S -passes=attributor --attributor-function-seed-allow-list asd < %s | FileCheck %s --check-prefixes=CHECK_DISABLED_FUNCTION

; RUN: opt -S -passes=attributor --attributor-function-seed-allow-list range_use1 < %s | FileCheck %s --check-prefixes=CHECK_ENABLED_FUNCTION




target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: nounwind uwtable
define internal i32 @range_test(i32 %a) #0 {
; CHECK_DISABLED: Function Attrs: noinline nounwind uwtable
; CHECK_DISABLED-LABEL: define {{[^@]+}}@range_test
; CHECK_DISABLED-SAME: (i32 [[A:%.*]])
; CHECK_DISABLED-NEXT:    [[TMP1:%.*]] = icmp sgt i32 [[A]], 100
; CHECK_DISABLED-NEXT:    [[TMP2:%.*]] = zext i1 [[TMP1]] to i32
; CHECK_DISABLED-NEXT:    ret i32 [[TMP2]]
;
; CHECK_DISABLED_FUNCTION: Function Attrs: noinline nounwind uwtable
; CHECK_DISABLED_FUNCTION-LABEL: define {{[^@]+}}@range_test
; CHECK_DISABLED_FUNCTION-SAME: (i32 [[A:%.*]])
; CHECK_DISABLED_FUNCTION-NEXT:    [[TMP1:%.*]] = icmp sgt i32 [[A]], 100
; CHECK_DISABLED_FUNCTION-NEXT:    [[TMP2:%.*]] = zext i1 [[TMP1]] to i32
; CHECK_DISABLED_FUNCTION-NEXT:    ret i32 [[TMP2]]
;
; CHECK_ENABLED_FUNCTION: Function Attrs: noinline nounwind readnone uwtable
; CHECK_ENABLED_FUNCTION-LABEL: define {{[^@]+}}@range_test()
; CHECK_ENABLED_FUNCTION-NEXT:    ret i32 1
;
  %1 = icmp sgt i32 %a, 100
  %2 = zext i1 %1 to i32
  ret i32 %2
}

; Function Attrs: nounwind uwtable
define i32 @range_use1() #0 {
; CHECK_DISABLED: Function Attrs: noinline nounwind uwtable
; CHECK_DISABLED-LABEL: define {{[^@]+}}@range_use1()
; CHECK_DISABLED-NEXT:    [[TMP1:%.*]] = call i32 @range_test(i32 123)
; CHECK_DISABLED-NEXT:    ret i32 [[TMP1]]
;
; CHECK_ENABLED: Function Attrs: noinline nounwind uwtable
; CHECK_ENABLED-LABEL: define {{[^@]+}}@range_use1()
; CHECK_ENABLED-NEXT:    ret i32 1
;
; CHECK_DISABLED_FUNCTION: Function Attrs: noinline nounwind uwtable
; CHECK_DISABLED_FUNCTION-LABEL: define {{[^@]+}}@range_use1()
; CHECK_DISABLED_FUNCTION-NEXT:    [[TMP1:%.*]] = call i32 @range_test(i32 123)
; CHECK_DISABLED_FUNCTION-NEXT:    ret i32 [[TMP1]]
;
; CHECK_ENABLED_FUNCTION: Function Attrs: nofree noinline nosync nounwind readnone uwtable willreturn
; CHECK_ENABLED_FUNCTION-LABEL: define {{[^@]+}}@range_use1()
; CHECK_ENABLED_FUNCTION-NEXT:    ret i32 1
;
  %1 = call i32 @range_test(i32 123)
  ret i32 %1
}

; Function Attrs: nounwind uwtable
define i32 @range_use2() #0 {
; CHECK_DISABLED: Function Attrs: noinline nounwind uwtable
; CHECK_DISABLED-LABEL: define {{[^@]+}}@range_use2()
; CHECK_DISABLED-NEXT:    [[TMP1:%.*]] = call i32 @range_test(i32 123)
; CHECK_DISABLED-NEXT:    ret i32 [[TMP1]]
;
; CHECK_ENABLED: Function Attrs: noinline nounwind uwtable
; CHECK_ENABLED-LABEL: define {{[^@]+}}@range_use2()
; CHECK_ENABLED-NEXT:    ret i32 1
;
; CHECK_DISABLED_FUNCTION: Function Attrs: noinline nounwind uwtable
; CHECK_DISABLED_FUNCTION-LABEL: define {{[^@]+}}@range_use2()
; CHECK_DISABLED_FUNCTION-NEXT:    [[TMP1:%.*]] = call i32 @range_test(i32 123)
; CHECK_DISABLED_FUNCTION-NEXT:    ret i32 [[TMP1]]
;
; CHECK_ENABLED_FUNCTION: Function Attrs: noinline nounwind uwtable
; CHECK_ENABLED_FUNCTION-LABEL: define {{[^@]+}}@range_use2()
; CHECK_ENABLED_FUNCTION-NEXT:    [[TMP1:%.*]] = call i32 @range_test()
; CHECK_ENABLED_FUNCTION-NEXT:    ret i32 [[TMP1]]
;
  %1 = call i32 @range_test(i32 123)
  ret i32 %1
}

attributes #0 = { nounwind uwtable noinline }

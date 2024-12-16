; RUN: rm -rf %t.rundir
; RUN: rm -rf %t.nchannel-basename.*
; RUN: mkdir %t.rundir
; RUN: cp %S/../../../../lib/Analysis/models/log_reader.py %t.rundir
; RUN: cp %S/../../../../lib/Analysis/models/interactive_host.py %t.rundir
; RUN: cp %S/Inputs/interactive_main.py %t.rundir

; RUN: %python %t.rundir/interactive_main.py 3 2.0 no %t.nchannel-basename \
; RUN:    opt %S/Inputs/nested.ll -S -O3 --mlgo-loop-unroll-interactive-channel-base=%t.nchannel-basename \
; RUN:    --mlgo-loop-unroll-advisor-mode=development \
; RUN:    --interactive-model-runner-echo-reply \
; RUN:    -debug-only=loop-unroll-development-advisor \
; RUN:    -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK1

; RUN: %python %t.rundir/interactive_main.py 3 0.5 no %t.nchannel-basename \
; RUN:    opt %S/Inputs/nested.ll -S -O3 --mlgo-loop-unroll-interactive-channel-base=%t.nchannel-basename \
; RUN:    --mlgo-loop-unroll-advisor-mode=development \
; RUN:    --interactive-model-runner-echo-reply \
; RUN:    -debug-only=loop-unroll-development-advisor \
; RUN:    -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK2

; RUN: rm -rf %t.rundir/instrumented-module.ll
; RUN: %python %t.rundir/interactive_main.py 3 2.5 instrument %t.nchannel-basename \
; RUN:    opt %S/Inputs/nested.ll -S -O3 --mlgo-loop-unroll-interactive-channel-base=%t.nchannel-basename \
; RUN:    --mlgo-loop-unroll-advisor-mode=development \
; RUN:    --interactive-model-runner-echo-reply \
; RUN:    -debug-only=loop-unroll-development-advisor \
; RUN:    -o %t.rundir/instrumented-module.ll -S
;
; RUN: cat %t.rundir/instrumented-module.ll | FileCheck %s --check-prefix=CHECK3

; CHECK1: mlgo-loop-unroll: got advice factor 5
; CHECK1: mlgo-loop-unroll: unattempted unroll
; CHECK1: mlgo-loop-unroll: got advice factor 5
; CHECK1: mlgo-loop-unroll: unattempted unroll
; CHECK1: mlgo-loop-unroll: got advice factor 5
; CHECK1: mlgo-loop-unroll: unrolled
; CHECK1: mlgo-loop-unroll: got advice factor 5
; CHECK1: mlgo-loop-unroll: unrolled

; CHECK2: mlgo-loop-unroll: got advice nounroll
; CHECK2: mlgo-loop-unroll: unattempted unroll
; CHECK2: mlgo-loop-unroll: got advice nounroll
; CHECK2: mlgo-loop-unroll: unattempted unroll
; CHECK2: mlgo-loop-unroll: got advice nounroll
; CHECK2: mlgo-loop-unroll: unattempted unroll
; CHECK2: mlgo-loop-unroll: got advice nounroll
; CHECK2: mlgo-loop-unroll: unattempted unroll

; CHECK3: test_loop_begin
; CHECK3: test_loop_end

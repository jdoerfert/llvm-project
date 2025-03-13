; RUN: opt -S --input-gen-mode=generate --passes=input-gen-instrument-entries,input-gen-instrument-memory %s | FileCheck %s


declare void @ext()

; We should not make changes to these globals
; CHECK: @_ZTIi = external constant ptr
; CHECK: declare i32 @__gxx_personality_v0(...)
@_ZTIi = external constant ptr
declare i32 @__gxx_personality_v0(...)

$group = comdat any
define i32 @group() uwtable comdat personality ptr @__gxx_personality_v0 {
entry:
  invoke void @ext() to label %try.cont unwind label %lpad
lpad:
  %0 = landingpad { ptr, i32 } catch ptr @_ZTIi
  br label %eh.resume
try.cont:
  ret i32 0
eh.resume:
  resume { ptr, i32 } %0
}

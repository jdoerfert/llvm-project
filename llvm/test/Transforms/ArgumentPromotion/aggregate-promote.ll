; RUN: opt < %s -argpromotion -S | FileCheck %s
; RUN: opt < %s -passes=argpromotion -S | FileCheck %s

%T = type { i32, i32, i32, i32 }
@G = constant %T { i32 0, i32 0, i32 17, i32 25 }

define internal i32 @test(%T* %p) {
; CHECK-LABEL: define internal i32 @test(
; CHECK: i32 %{{.*}}, i32 %{{.*}})
entry:
  %a.gep = getelementptr %T, %T* %p, i64 0, i32 3
  %b.gep = getelementptr %T, %T* %p, i64 0, i32 2
  %a = load i32, i32* %a.gep
  %b = load i32, i32* %b.gep
; CHECK-NOT: load
  %v = add i32 %a, %b
  ret i32 %v
; CHECK: ret i32
}

define internal i32 @bitcast1(%T* %p) {
; FIXME: This should look like: define internal i32 @bitcast1(i32 %{{.*}})
; CHECK-LABEL: define internal i32 @bitcast1(%T* %p)
entry:
; FIXME: The bitcast below is equivalent to the GEP
;   %bc = getelementptr %T, %T* %p, i64 0, i32 0
; but we currently fail to promote in the presence of bitcasts.
  %bc = bitcast %T* %p to i32*
  %v = load i32, i32* %bc
; FIXME: This should be a CHECK-NOT!
; CHECK: load
  ret i32 %v
; CHECK: ret i32
}

define internal i32 @bitcast2(%T* %p) {
; FIXME: This should look like: define internal i32 @bitcast2(i32 %{{.*}})
; CHECK-LABEL: define internal i32 @bitcast2(%T* %p)
entry:
; FIXME: The bitcast below is equivalent to the GEP
;   %gp = getelementptr %T, %T* %p, i64 0, i32 2
; but we currently fail to promote in the presence of bitcasts.
  %bc = bitcast %T* %p to i32*
  %gp = getelementptr i32, i32* %bc, i32 2
  %v = load i32, i32* %gp
; FIXME: This should be a CHECK-NOT!
; CHECK: load
  ret i32 %v
; CHECK: ret i32
}

define internal i32 @bitcast3(%T* %p) {
; FIXME: This should look like: define internal i32 @bitcast3(i32 %{{.*}})
; CHECK-LABEL: define internal i32 @bitcast3(%T* %p)
entry:
; FIXME: The bitcast below is equivalent to the GEP
;   %gp2 = getelementptr %T, %T* %p, i64 0, i32 3
; but we currently fail to promote in the presence of bitcasts.
  %gp1 = getelementptr %T, %T* %p, i64 0, i32 2
  %bc1 = bitcast i32* %gp1 to i8*
  %gp2 = getelementptr i8, i8* %bc1, i32 4
  %bc2 = bitcast i8* %gp2 to i32*
  %v = load i32, i32* %bc2
; FIXME: This should be a CHECK-NOT!
; CHECK: load
  ret i32 %v
; CHECK: ret i32
}

define i32 @caller() {
; CHECK-LABEL: define i32 @caller(
entry:
  %v1 = call i32 @test(%T* @G)
  %v2 = call i32 @bitcast1(%T* @G)
  %v3 = call i32 @bitcast2(%T* @G)
  %v4 = call i32 @bitcast3(%T* @G)
; CHECK-DAG: %[[B_GEP:.*]] = getelementptr %T, %T* @G, i64 0, i32 2
; CHECK-DAG: %[[B:.*]] = load i32, i32* %[[B_GEP]]
; CHECK-DAG: %[[A_GEP:.*]] = getelementptr %T, %T* @G, i64 0, i32 3
; CHECK-DAG: %[[A:.*]] = load i32, i32* %[[A_GEP]]
; CHECK: call i32 @test(i32 %[[B]], i32 %[[A]])
; FIXME: This should look like: 
;   %[[BC1_GEP:.*]] = getelementptr %T, %T* @G, i64 0, i32 0
;   %[[BC1_V:.*]] = load i32, i32* %[[BC1_GEP]]
;   call i32 @bitcast1(i32 %[[BC1_V]])
; CHECK: call i32 @bitcast1(%T* @G)
; FIXME: This should look like: 
;   %[[BC2_GEP:.*]] = getelementptr %T, %T* @G, i64 0, i32 2
;   %[[BC2_V:.*]] = load i32, i32* %[[BC2_GEP]]
;   call i32 @bitcast2(i32 %[[BC2_V]])
; CHECK: call i32 @bitcast2(%T* @G)
; FIXME: This should look like: 
;   %[[BC3_GEP:.*]] = getelementptr %T, %T* @G, i64 0, i32 3
;   %[[BC3_V:.*]] = load i32, i32* %[[BC3_GEP]]
;   call i32 @bitcast3(i32 %[[BC3_V]])
; CHECK: call i32 @bitcast3(%T* @G)
  %add1 = add i32 %v1, %v2
  %add2 = add i32 %v3, %v4
  %mul = mul i32 %add1, %add2
  ret i32 %mul
}

; RUN: opt < %s -argpromotion -S | FileCheck %s
; RUN: opt < %s -passes=argpromotion -S | FileCheck %s

declare void @sink(i32)

; CHECK: define internal void @test({{.*}} !dbg [[SP:![0-9]+]]
define internal void @test(i32** %X) !dbg !2 {
  %1 = load i32*, i32** %X, align 8
  %2 = load i32, i32* %1, align 8
  call void @sink(i32 %2)
  ret void
}

%struct.pair = type { i32, i32 }

; CHECK: define internal i32 @test_byval(i32 %{{.*}}, i32 %{{.*}})
define internal i32 @test_byval(%struct.pair* byval %P) {
  %g = getelementptr %struct.pair, %struct.pair* %P, i32 0, i32 0
  %v = load i32, i32* %g, align 8
  ret i32 %v
}

; Make sure unused byval arguments are not promoted but removed
;
; FIXME: This should be: define internal void @test_byval_2()
; Related to PR42852
; CHECK: define internal void @test_byval_2(i32 %{{.*}}, i32 %{{.*}})
define internal void @test_byval_2(%struct.pair* byval %P) {
  ret void
}

; Make sure unused byval arguments are not promoted but removed (or kept)
;
; CHECK: define internal void @test_byval_2()
define internal void @test_byval_2(%struct.pair* byval %P) {
  ret void
}

; CHECK-LABEL: define {{.*}} @caller(
define i32 @caller(i32** %Y, %struct.pair* %P) {
; CHECK:  load i32*, {{.*}} !dbg [[LOC_1:![0-9]+]]
; CHECK-NEXT:  load i32, {{.*}} !dbg [[LOC_1]]
; CHECK-NEXT: call void @test(i32 %{{.*}}), !dbg [[LOC_1]]
  call void @test(i32** %Y), !dbg !1

; CHECK: getelementptr %struct.pair, {{.*}} !dbg [[LOC_2:![0-9]+]]
; CHECK-NEXT: load i32, i32* {{.*}} !dbg [[LOC_2]]
; CHECK-NEXT: getelementptr %struct.pair, {{.*}} !dbg [[LOC_2]]
; CHECK-NEXT: load i32, i32* {{.*}} !dbg [[LOC_2]]
; CHECK-NEXT: call i32 @test_byval(i32 %{{.*}}, i32 %{{.*}}), !dbg [[LOC_2]]
  %v = call i32 @test_byval(%struct.pair* %P), !dbg !6
; CHECK: call void @test_byval_2(), !dbg [[LOC_2:![0-9]+]]
  call void @test_byval_2(%struct.pair* %P), !dbg !6
  ret i32 %v
}

; CHECK: [[SP]] = distinct !DISubprogram(name: "test",
; CHECK: [[LOC_1]] = !DILocation(line: 8
; CHECK: [[LOC_2]] = !DILocation(line: 9

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!3}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !DILocation(line: 8, scope: !2)
!2 = distinct !DISubprogram(name: "test", file: !5, line: 3, isLocal: true, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !3, scopeLine: 3, scope: null)
!3 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.5.0 ", isOptimized: false, emissionKind: LineTablesOnly, file: !5)
!5 = !DIFile(filename: "test.c", directory: "")
!6 = !DILocation(line: 9, scope: !2)

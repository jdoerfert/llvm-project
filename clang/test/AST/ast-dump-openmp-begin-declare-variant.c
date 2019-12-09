// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -ast-dump %s | FileCheck %s

int also_before(void) {
  return 0;
}

#pragma omp begin declare variant match(device={kind(cpu)})
int also_after(void) {
  return 1;
}
int also_before(void) {
  return 1;
}
#pragma omp end declare variant

#pragma omp begin declare variant match(device={kind(gpu)})
int also_after(void) {
  return 2;
}
int also_before(void) {
  return 2;
}
#pragma omp end declare variant


#pragma omp begin declare variant match(device={kind(fpga)})

This text is never parsed!

#pragma omp end declare variant

int also_after(void) {
  return 0;
}

int test() {
  return also_after() + also_before();
}

// Make sure:
// 1) we pick the right versions, that is test should reference the kind(cpu) versions.
// 2) we do not see the ast nodes for the gpu kind
// 3) we do not choke on the text in the kind(fpga) guarded scope.

// CHECK:       -FunctionDecl {{.*}} <{{.*}}3:1, line:{{.*}}:1> line:{{.*}}:5 also_before 'int (void)'
// CHECK-NEXT:  | |-CompoundStmt {{.*}} <col:23, line:{{.*}}:1>
// CHECK-NEXT:  | | `-ReturnStmt {{.*}} <line:{{.*}}:3, col:10>
// CHECK-NEXT:  | |   `-IntegerLiteral {{.*}} <col:10> 'int' 0
// CHECK-NEXT:  | `-OMPDeclareVariantAttr {{.*}} <line:{{.*}}:1, col:60> Inherited Implicit 1 1 cpu
// CHECK-NEXT:  |   |-<<<NULL>>>
// CHECK-NEXT:  |   `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
// CHECK-NEXT:  |-FunctionDecl [[GOOD_ALSO_AFTER:0x[a-z0-9]*]] <line:{{.*}}:1, line:{{.*}}:1> line:{{.*}}:5 used also_after 'int (void)'
// CHECK-NEXT:  | |-CompoundStmt {{.*}} <col:22, line:{{.*}}:1>
// CHECK-NEXT:  | | `-ReturnStmt {{.*}} <line:{{.*}}:3, col:10>
// CHECK-NEXT:  | |   `-IntegerLiteral {{.*}} <col:10> 'int' 1
// CHECK-NEXT:  | `-OMPDeclareVariantAttr {{.*}} <line:{{.*}}:1, col:60> Implicit 1 1 cpu
// CHECK-NEXT:  |   |-<<<NULL>>>
// CHECK-NEXT:  |   `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
// CHECK-NEXT:  |-FunctionDecl [[GOOD_ALSO_BEFORE:0x[a-z0-9]*]] <line:{{.*}}:1, line:{{.*}}:1> line:{{.*}}:5 used also_before 'int (void)'
// CHECK-NEXT:  | |-CompoundStmt {{.*}} <col:23, line:{{.*}}:1>
// CHECK-NEXT:  | | `-ReturnStmt {{.*}} <line:{{.*}}:3, col:10>
// CHECK-NEXT:  | |   `-IntegerLiteral {{.*}} <col:10> 'int' 1
// CHECK-NEXT:  | `-OMPDeclareVariantAttr {{.*}} <line:{{.*}}:1, col:60> Implicit 1 1 cpu
// CHECK-NEXT:  |   |-<<<NULL>>>
// CHECK-NEXT:  |   `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
// CHECK-NEXT:  |-FunctionDecl {{.*}} <line:{{.*}}:1, line:{{.*}}:1> line:{{.*}}:5 also_after 'int (void)'
// CHECK-NEXT:  | |-CompoundStmt {{.*}} <col:22, line:{{.*}}:1>
// CHECK-NEXT:  | | `-ReturnStmt {{.*}} <line:{{.*}}:3, col:10>
// CHECK-NEXT:  | |   `-IntegerLiteral {{.*}} <col:10> 'int' 0
// CHECK-NEXT:  | `-OMPDeclareVariantAttr {{.*}} <line:{{.*}}:1, col:60> Inherited Implicit 1 1 cpu
// CHECK-NEXT:  |   |-<<<NULL>>>
// CHECK-NEXT:  |   `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
// CHECK-NEXT:  `-FunctionDecl {{.*}} <line:{{.*}}:1, line:{{.*}}:1> line:{{.*}}:5 test 'int ()'
// CHECK-NEXT:    `-CompoundStmt {{.*}} <col:12, line:{{.*}}:1>
// CHECK-NEXT:      `-ReturnStmt {{.*}} <line:{{.*}}:3, col:37>
// CHECK-NEXT:        `-BinaryOperator {{.*}} <col:10, col:37> 'int' '+'
// CHECK-NEXT:          |-CallExpr {{.*}} <col:10, col:21> 'int'
// CHECK-NEXT:          | `-ImplicitCastExpr {{.*}} <col:10> 'int (*)(void)' <FunctionToPointerDecay>
// CHECK-NEXT:          |   `-DeclRefExpr {{.*}} <col:10> 'int (void)' lvalue Function [[GOOD_ALSO_AFTER]] 'also_after' 'int (void)'
// CHECK-NEXT:          `-CallExpr {{.*}} <col:25, col:37> 'int'
// CHECK-NEXT:            `-ImplicitCastExpr {{.*}} <col:25> 'int (*)(void)' <FunctionToPointerDecay>
// CHECK-NEXT:              `-DeclRefExpr {{.*}} <col:25> 'int (void)' lvalue Function [[GOOD_ALSO_BEFORE]] 'also_before' 'int (void)'


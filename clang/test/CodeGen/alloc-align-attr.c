// RUN: %clang_cc1 -triple x86_64-pc-linux -emit-llvm -o - %s | FileCheck %s

__INT32_TYPE__*m1(__INT32_TYPE__ i) __attribute__((alloc_align(1)));

// Condition where parameter to m1 is not size_t.
__INT32_TYPE__ test1(__INT32_TYPE__ a) {
// CHECK: define i32 @test1
  return *m1(a);
// CHECK: [[CALL1:%.+]] = call i32* @m1(i32 [[PARAM1:%[^\)]+]])
// CHECK: [[ALIGNCAST1:%.+]] = zext i32 [[PARAM1]] to i64
// CHECK: call void @llvm.assume(i1 true) [ "align"(i32* [[CALL1]], i64 [[ALIGNCAST1]]) ]
}
// Condition where test2 param needs casting.
__INT32_TYPE__ test2(__SIZE_TYPE__ a) {
// CHECK: define i32 @test2
  return *m1(a);
// CHECK: [[CONV2:%.+]] = trunc i64 %{{.+}} to i32
// CHECK: [[CALL2:%.+]] = call i32* @m1(i32 [[CONV2]])
// CHECK: [[ALIGNCAST2:%.+]] = zext i32 [[CONV2]] to i64
// CHECK: call void @llvm.assume(i1 true) [ "align"(i32* [[CALL2]], i64 [[ALIGNCAST2]]) ]
}
__INT32_TYPE__ *m2(__SIZE_TYPE__ i) __attribute__((alloc_align(1)));

// test3 param needs casting, but 'm2' is correct.
__INT32_TYPE__ test3(__INT32_TYPE__ a) {
// CHECK: define i32 @test3
  return *m2(a);
// CHECK: [[CONV3:%.+]] = sext i32 %{{.+}} to i64
// CHECK: [[CALL3:%.+]] = call i32* @m2(i64 [[CONV3]])
// CHECK: call void @llvm.assume(i1 true) [ "align"(i32* [[CALL3]], i64 [[CONV3]]) ]
}

// Every type matches, canonical example.
__INT32_TYPE__ test4(__SIZE_TYPE__ a) {
// CHECK: define i32 @test4
  return *m2(a);
// CHECK: [[CALL4:%.+]] = call i32* @m2(i64 [[PARAM4:%[^\)]+]])
// CHECK: call void @llvm.assume(i1 true) [ "align"(i32* [[CALL4]], i64 [[PARAM4]]) ]
}


struct Empty {};
struct MultiArgs { __INT64_TYPE__ a, b;};
// Struct parameter doesn't take up an IR parameter, 'i' takes up 2.
// Truncation to i64 is permissible, since alignments of greater than 2^64 are insane.
__INT32_TYPE__ *m3(struct Empty s, __int128_t i) __attribute__((alloc_align(2)));
__INT32_TYPE__ test5(__int128_t a) {
// CHECK: define i32 @test5
  struct Empty e;
  return *m3(e, a);
// CHECK: [[CALL5:%.+]] = call i32* @m3(i64 %{{.*}}, i64 %{{.*}})
// CHECK: [[ALIGNCAST5:%.+]] = trunc i128 %{{.*}} to i64
// CHECK-NEXT: call void @llvm.assume(i1 true) [ "align"(i32* [[CALL5]], i64 [[ALIGNCAST5]]) ]
}
// Struct parameter takes up 2 parameters, 'i' takes up 2.
__INT32_TYPE__ *m4(struct MultiArgs s, __int128_t i) __attribute__((alloc_align(2)));
__INT32_TYPE__ test6(__int128_t a) {
// CHECK: define i32 @test6
  struct MultiArgs e;
  return *m4(e, a);
// CHECK: [[CALL6:%.+]] = call i32* @m4(i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
// CHECK-NEXT: [[ALIGNCAST6:%.+]] = trunc i128 %{{.*}} to i64
// CHECK-NEXT: call void @llvm.assume(i1 true) [ "align"(i32* [[CALL6]], i64 [[ALIGNCAST6]]) ]
}


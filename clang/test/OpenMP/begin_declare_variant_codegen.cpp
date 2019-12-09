// RUN: %clang_cc1 -verify -fopenmp -x c++ -emit-llvm %s -triple %itanium_abi_triple -fexceptions -fcxx-exceptions -o - | FileCheck %s
// expected-no-diagnostics

int bar(void) {
  return 0;
}

template <typename T>
T baz(void) { return 0; }

#pragma omp begin declare variant match(device={kind(cpu)})
int foo(void) {
  return 1;
}
int bar(void) {
  return 1;
}
template <typename T>
T baz(void) { return 1; }

template <typename T>
T biz(void) { return 1; }

template <typename T>
T buz(void) { return 3; }

template <>
char buz(void) { return 1; }

template <typename T>
T bez(void) { return 3; }
#pragma omp end declare variant

#pragma omp begin declare variant match(device={kind(gpu)})
int foo(void) {
  return 2;
}
int bar(void) {
  return 2;
}
#pragma omp end declare variant


#pragma omp begin declare variant match(device={kind(fpga)})

This text is never parsed!

#pragma omp end declare variant

int foo(void) {
  return 0;
}

template <typename T>
T biz(void) { return 0; }

template <>
char buz(void) { return 0; }

template <>
long bez(void) { return 0; }

#pragma omp begin declare variant match(device = {kind(cpu)})
template <>
long bez(void) { return 1; }
#pragma omp end declare variant

int test() {
  return foo() + bar() + baz<int>() + biz<short>() + buz<char>() + bez<long>();
}

// Make sure all ompvariant functions return 1 and all others return 0.

// CHECK:       ; Function Attrs:
// CHECK-NEXT:  define i32 @_Z3barv()
// CHECK-NEXT:  entry:
// CHECK-NEXT:    ret i32 0
// CHECK-NEXT:  }
// CHECK:       ; Function Attrs:
// CHECK-NEXT:  define i32 @_Z3foov.ompvariant()
// CHECK-NEXT:  entry:
// CHECK-NEXT:    ret i32 1
// CHECK-NEXT:  }
// CHECK:       ; Function Attrs:
// CHECK-NEXT:  define i32 @_Z3barv.ompvariant()
// CHECK-NEXT:  entry:
// CHECK-NEXT:    ret i32 1
// CHECK-NEXT:  }
// CHECK:       ; Function Attrs:
// CHECK-NEXT:  define signext i8 @_Z3buzIcET_v.ompvariant()
// CHECK-NEXT:  entry:
// CHECK-NEXT:    ret i8 1
// CHECK-NEXT:  }
// CHECK:       ; Function Attrs:
// CHECK-NEXT:  define i32 @_Z3foov()
// CHECK-NEXT:  entry:
// CHECK-NEXT:    ret i32 0
// CHECK-NEXT:  }
// CHECK:       ; Function Attrs:
// CHECK-NEXT:  define signext i8 @_Z3buzIcET_v()
// CHECK-NEXT:  entry:
// CHECK-NEXT:    ret i8 0
// CHECK-NEXT:  }
// CHECK:       ; Function Attrs:
// CHECK-NEXT:  define i64 @_Z3bezIlET_v()
// CHECK-NEXT:  entry:
// CHECK-NEXT:    ret i64 0
// CHECK-NEXT:  }
// CHECK:       ; Function Attrs:
// CHECK-NEXT:  define i64 @_Z3bezIlET_v.ompvariant()
// CHECK-NEXT:  entry:
// CHECK-NEXT:    ret i64 1
// CHECK-NEXT:  }

// Make sure we call only ompvariant functions

// CHECK:  define i32 @_Z4testv()
// CHECK:    %call = call i32 @_Z3foov.ompvariant()
// CHECK:    %call1 = call i32 @_Z3barv.ompvariant()
// CHECK:    %call2 = call i32 @_Z3bazIiET_v.ompvariant()
// CHECK:    %call4 = call signext i16 @_Z3bizIsET_v.ompvariant()
// CHECK:    %call6 = call signext i8 @_Z3buzIcET_v.ompvariant()
// CHECK:    %call10 = call i64 @_Z3bezIlET_v.ompvariant()

// CHECK:       ; Function Attrs:
// CHECK-NEXT:  define linkonce_odr i32 @_Z3bazIiET_v.ompvariant()
// CHECK-NEXT:  entry:
// CHECK-NEXT:    ret i32 1
// CHECK-NEXT:  }
// CHECK:       ; Function Attrs:
// CHECK-NEXT:  define linkonce_odr signext i16 @_Z3bizIsET_v.ompvariant()
// CHECK-NEXT:  entry:
// CHECK-NEXT:    ret i16 1
// CHECK-NEXT:  }

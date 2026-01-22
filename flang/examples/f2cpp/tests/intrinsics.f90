! RUN: echo '#include "f2cpp_rt.h"' > %t.cpp
! RUN: echo '' >> %t.cpp
! RUN: %flang_fc1 -load %llvmshlibdir/flang2cpp%pluginext -plugin f2cpp %s >> %t.cpp
! RUN: %clangxx %t.cpp -I%S/../include -L%llvmshlibdir -lf2cpp_rt -lflang_rt.runtime -o %t
! RUN: %t | FileCheck %s

! Test Intrinsic Functions
program intrinsics
  character(len=10) :: str
  integer :: l
  str = "Hello"
  l = len(str)
  print *, "len(str) =", l
  print *, "trim(str) = ", trim(str)
end program intrinsics

! CHECK: len(str) = 10
! CHECK: trim(str) = Hello

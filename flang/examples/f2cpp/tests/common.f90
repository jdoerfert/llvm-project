! RUN: echo '#include "f2cpp_rt.h"' > %t.cpp
! RUN: echo '#include "%{t:stem}.tmp.h"' >> %t.cpp
! RUN: echo '#include "%{t:stem}.tmp.local.h"' >> %t.cpp
! RUN: echo '' >> %t.cpp
! RUN: %flang_fc1 -load %llvmshlibdir/flang2cpp%pluginext -plugin f2cpp -mllvm -declarations="%t.h" -mllvm -local-declarations="%t.local.h" -mllvm -common-blocks="%t.common.c" %s >> %t.cpp
! RUN: %clang %t.common.c -I%S/../include -c -o %t.common.o
! RUN: %clangxx %t.cpp -I%S/../include -L%llvmshlibdir -lf2cpp_rt -lflang_rt.runtime %t.common.o -o %t
! RUN: %t | FileCheck %s

! Test Common Block
program common_test
  integer :: a, b
  common /myblock/ a, b
  a = 10
  b = 20
  call print_common()
end program common_test

subroutine print_common()
  integer :: x, y
  common /myblock/ x, y
  print *, "Common:", x, y
end subroutine print_common

! CHECK: Common: 10 20

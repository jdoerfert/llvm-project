! RUN: echo '#include "f2cpp_rt.h"' > %t.cpp
! RUN: echo '' >> %t.cpp
! RUN: %flang_fc1 -load %llvmshlibdir/flang2cpp%pluginext -plugin f2cpp %s >> %t.cpp
! RUN: %clangxx %t.cpp -I%S/../include -L%llvmshlibdir -lf2cpp_rt -lflang_rt.runtime -o %t
! RUN: %t | FileCheck %s

! Test real arithmetic
program real_arithmetic
  real :: a, b
  a = 4.0
  b = 2.0
  print *, "Add", a + b
  print *, "Sub", a - b
  print *, "Mul", a * b
  print *, "Div", a / b
end program real_arithmetic

! CHECK: Add 6.
! CHECK: Sub 2.
! CHECK: Mul 8.
! CHECK: Div 2.

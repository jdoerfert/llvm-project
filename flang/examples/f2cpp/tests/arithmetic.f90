! RUN: echo '#include "f2cpp_rt.h"' > %t.cpp
! RUN: echo '' >> %t.cpp
! RUN: %flang_fc1 -load %llvmshlibdir/flang2cpp%pluginext -plugin f2cpp %s >> %t.cpp
! RUN: %clangxx %t.cpp -I%S/../include -L%llvmshlibdir -lf2cpp_rt -lflang_rt.runtime -o %t
! RUN: %t | FileCheck %s

! Test Integer Arithmetic
program arithmetic
  integer :: a, b, c
  a = 10
  b = 2
  print *, "a =", a
  print *, "b =", b
  c = a + b
  print *, "a + b =", c
  c = a - b
  print *, "a - b =", c
  c = a * b
  print *, "a * b =", c
  c = a / b
  print *, "a / b =", c
end program arithmetic

! CHECK: a = 10
! CHECK: b = 2
! CHECK: a + b = 12
! CHECK: a - b = 8
! CHECK: a * b = 20
! CHECK: a / b = 5

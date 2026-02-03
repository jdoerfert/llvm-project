! RUN: echo '#include "f2cpp_rt.h"' > %t.cpp
! RUN: echo '#include "%t.h"' >> %t.cpp
! RUN: echo '' >> %t.cpp
! RUN: rm %t.h
! RUN: %flang_fc1 -load %llvmshlibdir/flang2cpp%pluginext -plugin f2cpp -mllvm -declarations=%t.h %s >> %t.cpp
! RUN: %clangxx %t.cpp -I%S/../include -L%llvmshlibdir -lf2cpp_rt -lflang_rt.runtime -o %t
! RUN: %t | FileCheck %s

! Test Intent
program intent_test
  integer :: a, b
  a = 10
  call update(a, b)
  print *, "a =", a, "b =", b
end program intent_test

subroutine update(x, y)
  integer, intent(in) :: x
  integer, intent(out) :: y
  y = x * 2
end subroutine update

! CHECK: a = 10 b = 20

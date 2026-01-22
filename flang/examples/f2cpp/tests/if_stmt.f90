! RUN: echo '#include "f2cpp_rt.h"' > %t.cpp
! RUN: echo '' >> %t.cpp
! RUN: %flang_fc1 -load %llvmshlibdir/flang2cpp%pluginext -plugin f2cpp %s >> %t.cpp
! RUN: %clangxx %t.cpp -I%S/../include -L%llvmshlibdir -lf2cpp_rt -lflang_rt.runtime -o %t
! RUN: %t | FileCheck %s

! Test IF Statements
program if_stmt
  integer :: a
  a = 10
  if (a > 5) then
    print *, "a > 5"
  end if
  if (a < 5) then
    print *, "a < 5"
  else
    print *, "a >= 5"
  end if
  if (a == 10) then
    print *, "a == 10"
  else if (a == 20) then
    print *, "a == 20"
  else
    print *, "a != 10 and a != 20"
  end if
end program if_stmt

! CHECK: a > 5
! CHECK: a >= 5
! CHECK: a == 10

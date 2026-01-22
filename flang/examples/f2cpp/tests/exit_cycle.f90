! RUN: echo '#include "f2cpp_rt.h"' > %t.cpp
! RUN: echo '' >> %t.cpp
! RUN: %flang_fc1 -load %llvmshlibdir/flang2cpp%pluginext -plugin f2cpp %s >> %t.cpp
! RUN: %clangxx %t.cpp -I%S/../include -L%llvmshlibdir -lf2cpp_rt -lflang_rt.runtime -o %t
! RUN: %t | FileCheck %s

! Test exit and cycle statements
program exit_cycle
  integer :: i
  do i = 1, 10
    if (i == 5) cycle
    if (i == 8) exit
    print *, i
  end do
end program exit_cycle

! CHECK: 1
! CHECK: 2
! CHECK: 3
! CHECK: 4
! CHECK: 6
! CHECK: 7

! RUN: echo '#include "f2cpp_rt.h"' > %t.cpp
! RUN: echo '' >> %t.cpp
! RUN: %flang_fc1 -load %llvmshlibdir/flang2cpp%pluginext -plugin f2cpp %s >> %t.cpp
! RUN: %clangxx %t.cpp -I%S/../include -L%llvmshlibdir -lf2cpp_rt -lflang_rt.runtime -o %t
! RUN: %t | FileCheck %s

! Test Select Case
program select_case
  integer :: i
  i = 2
  select case (i)
  case (1)
    print *, "One"
  case (2)
    print *, "Two"
  case (3)
    print *, "Three"
  case default
    print *, "Other"
  end select
end program select_case

! CHECK: Two

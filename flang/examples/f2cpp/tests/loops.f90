! RUN: echo '#include "f2cpp_rt.h"' > %t.cpp
! RUN: echo '' >> %t.cpp
! RUN: %flang_fc1 -load %llvmshlibdir/flang2cpp%pluginext -plugin f2cpp %s >> %t.cpp
! RUN: %clangxx %t.cpp -I%S/../include -L%llvmshlibdir -lf2cpp_rt -lflang_rt.runtime -o %t
! RUN: %t | FileCheck %s

! Test DO Loops
program loops
  integer :: i, sum
  sum = 0
  ! Simple loop
  do i = 1, 5
    sum = sum + i
  end do
  print *, "Sum 1-5 =", sum

  ! Step loop
  sum = 0
  do i = 1, 10, 2
    sum = sum + i
  end do
  print *, "Sum odds 1-10 =", sum
end program loops

! CHECK: Sum 1-5 = 15
! CHECK: Sum odds 1-10 = 25

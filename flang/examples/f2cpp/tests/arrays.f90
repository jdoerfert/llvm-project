! RUN: echo '#include "f2cpp_rt.h"' > %t.cpp
! RUN: echo '' >> %t.cpp
! RUN: %flang_fc1 -load %llvmshlibdir/flang2cpp%pluginext -plugin f2cpp %s >> %t.cpp
! RUN: %clangxx %t.cpp -I%S/../include -L%llvmshlibdir -lf2cpp_rt -lflang_rt.runtime -o %t
! RUN: %t | FileCheck %s

! Test Static Arrays
program arrays
  integer, dimension(3) :: a
  integer :: i
  a(1) = 10
  a(2) = 20
  a(3) = 30
  print *, "a(1) =", a(1)
  print *, "a(2) =", a(2)
  print *, "a(3) =", a(3)
  do i = 1, 3
    print *, "loop a(", i, ") =", a(i)
  end do
end program arrays

! CHECK: a(1) = 10
! CHECK: a(2) = 20
! CHECK: a(3) = 30
! CHECK: loop a( 1 ) = 10
! CHECK: loop a( 2 ) = 20
! CHECK: loop a( 3 ) = 30

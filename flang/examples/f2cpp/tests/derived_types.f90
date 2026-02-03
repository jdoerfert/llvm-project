! RUN: echo '#include "f2cpp_rt.h"' > %t.cpp
! RUN: echo '' >> %t.cpp
! RUN: %flang_fc1 -load %llvmshlibdir/flang2cpp%pluginext -plugin f2cpp %s >> %t.cpp
! RUN: %clangxx %t.cpp -I%S/../include -L%llvmshlibdir -lf2cpp_rt -lflang_rt.runtime -o %t
! RUN: %t | FileCheck %s

! Test Derived Types
program derived_types
  type :: point
    integer :: x, y
  end type point

  type(point) :: p
  p%x = 10
  p%y = 20
  print *, "p =", p%x, p%y
end program derived_types

! CHECK: p = 10 20

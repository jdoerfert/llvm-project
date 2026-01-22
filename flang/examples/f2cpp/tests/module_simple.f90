! RUN: echo '#include "f2cpp_rt.h"' > %t.cpp
! RUN: echo '' >> %t.cpp
! RUN: %flang_fc1 -load %llvmshlibdir/flang2cpp%pluginext -plugin f2cpp %s >> %t.cpp
! RUN: %clangxx %t.cpp -I%S/../include -L%llvmshlibdir -lf2cpp_rt -lflang_rt.runtime -o %t
! RUN: %t | FileCheck %s

! Test simple module
module my_mod
  integer :: x = 10
contains
  subroutine print_x()
    print *, "x =", x
  end subroutine
end module my_mod

program module_simple
  use my_mod
  call print_x()
  x = 20
  call print_x()
end program module_simple

! CHECK: x = 10
! CHECK: x = 20

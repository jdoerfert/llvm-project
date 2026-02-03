! RUN: echo '#include "f2cpp_rt.h"' > %t.cpp
! RUN: echo '' >> %t.cpp
! RUN: %flang_fc1 -load %llvmshlibdir/flang2cpp%pluginext -plugin f2cpp %s >> %t.cpp
! RUN: %clangxx %t.cpp -I%S/../include -L%llvmshlibdir -lf2cpp_rt -lflang_rt.runtime -o %t
! RUN: %t | FileCheck %s

! Test Interface
program interface_test
  interface
    subroutine print_val(x)
      integer, intent(in) :: x
    end subroutine print_val
  end interface
  
  call print_val(42)
end program interface_test

subroutine print_val(x)
  integer, intent(in) :: x
  print *, "Val =", x
end subroutine print_val

! CHECK: Val = 42

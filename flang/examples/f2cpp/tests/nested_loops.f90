! RUN: echo '#include "f2cpp_rt.h"' > %t.cpp
! RUN: echo '' >> %t.cpp
! RUN: %flang_fc1 -load %llvmshlibdir/flang2cpp%pluginext -plugin f2cpp %s >> %t.cpp
! RUN: %clangxx %t.cpp -I%S/../include -L%llvmshlibdir -lf2cpp_rt -lflang_rt.runtime -o %t
! RUN: %t | FileCheck %s

! Test nested loops
program nested_loops
  integer :: i, j
  
  do i = 1, 3
    do j = 1, 2
      print *, "i=", i, "j=", j
    end do
  end do
end program nested_loops

! CHECK: i= 1 j= 1
! CHECK: i= 1 j= 2
! CHECK: i= 2 j= 1
! CHECK: i= 2 j= 2
! CHECK: i= 3 j= 1
! CHECK: i= 3 j= 2

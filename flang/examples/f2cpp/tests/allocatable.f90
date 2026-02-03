! RUN: echo '#include "f2cpp_rt.h"' > %t.cpp
! RUN: echo '' >> %t.cpp
! RUN: %flang_fc1 -load %llvmshlibdir/flang2cpp%pluginext -plugin f2cpp %s >> %t.cpp
! RUN: %clangxx %t.cpp -I%S/../include -L%llvmshlibdir -lf2cpp_rt -lflang_rt.runtime -o %t
! RUN: %t | FileCheck %s

! Test Allocatable Arrays
program allocatable_test
  integer, allocatable :: arr(:)
  allocate(arr(3))
  arr(1) = 1
  arr(2) = 2
  arr(3) = 3
  print *, "size =", size(arr)
  print *, "arr =", arr(1), arr(2), arr(3)
  deallocate(arr)
end program allocatable_test

! CHECK: size = 3
! CHECK: arr = 1 2 3

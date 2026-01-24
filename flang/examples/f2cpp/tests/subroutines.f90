! RUN: echo '#include "f2cpp_rt.h"' > %t.cpp
! RUN: echo '#include "%{t:stem}.tmp.h"' >> %t.cpp
! RUN: echo '' >> %t.cpp
! RUN: %flang_fc1 -load %llvmshlibdir/flang2cpp%pluginext -plugin f2cpp -mllvm -declarations="%t.h" %s >> %t.cpp
! RUN: %clangxx %t.cpp -I%S/../include -L%llvmshlibdir -lf2cpp_rt -lflang_rt.runtime -o %t
! RUN: %t | FileCheck %s

! Test subroutines
program subroutines
  call my_sub(10)
  call my_sub(20)
end program subroutines

subroutine my_sub(n)
  integer :: n
  print *, "Value is", n
end subroutine my_sub

! CHECK: Value is 10
! CHECK: Value is 20

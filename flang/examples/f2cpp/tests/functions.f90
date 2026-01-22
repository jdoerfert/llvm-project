! RUN: echo '#include "f2cpp_rt.h"' > %t.cpp
! RUN: echo '#include "%{t:stem}.tmp.h"' >> %t.cpp
! RUN: echo '' >> %t.cpp
! RUN: %flang_fc1 -load %llvmshlibdir/flang2cpp%pluginext -plugin f2cpp -mllvm -function-declarations="%t.h" %s >> %t.cpp
! RUN: %clangxx %t.cpp -I%S/../include -L%llvmshlibdir -lf2cpp_rt -lflang_rt.runtime -o %t
! RUN: %t | FileCheck %s

! Test Functions
program functions
  integer :: res
  res = add(10, 20)
  print *, "10 + 20 =", res
end program functions

integer function add(a, b)
  integer :: a, b
  add = a + b
end function add

! CHECK: 10 + 20 = 30

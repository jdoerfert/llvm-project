! RUN: echo '#include "f2cpp_rt.h"' > %t.cpp
! RUN: echo '#include "%t.h"' >> %t.cpp
! RUN: echo '' >> %t.cpp
! RUN: %flang_fc1 -load %llvmshlibdir/flang2cpp%pluginext -plugin f2cpp -mllvm -declarations="%t.h" %s >> %t.cpp
! RUN: %clangxx %t.cpp -I%S/../include -L%llvmshlibdir -lf2cpp_rt -lflang_rt.runtime -o %t
! RUN: %t | FileCheck %s

! Test Recursion
program recursion
  integer :: res
  res = factorial(5)
  print *, "5! =", res
end program recursion

recursive function factorial(n) result(res)
  integer :: n, res
  if (n <= 1) then
    res = 1
  else
    res = n * factorial(n - 1)
  end if
end function factorial

! CHECK: 5! = 120
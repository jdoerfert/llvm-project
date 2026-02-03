! RUN: echo '#include "f2cpp_rt.h"' > %t.cpp
! RUN: echo '' >> %t.cpp
! RUN: %flang_fc1 -load %llvmshlibdir/flang2cpp%pluginext -plugin f2cpp %s >> %t.cpp
! RUN: %clangxx %t.cpp -I%S/../include -L%llvmshlibdir -lf2cpp_rt -lflang_rt.runtime -o %t
! RUN: %t | FileCheck %s

! Test Pointers
program pointers
  integer, target :: t
  integer, pointer :: p
  t = 10
  p => t
  print *, "p points to", p
  p = 20
  print *, "t is now", t
end program pointers

! CHECK: p points to 10
! CHECK: t is now 20

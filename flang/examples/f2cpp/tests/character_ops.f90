! RUN: echo '#include "f2cpp_rt.h"' > %t.cpp
! RUN: echo '' >> %t.cpp
! RUN: %flang_fc1 -load %llvmshlibdir/flang2cpp%pluginext -plugin f2cpp %s >> %t.cpp
! RUN: %clangxx %t.cpp -I%S/../include -L%llvmshlibdir -lf2cpp_rt -lflang_rt.runtime -o %t
! RUN: %t | FileCheck %s

! Test Character Operations
program character_ops
  character(len=5) :: s1
  character(len=6) :: s2
  character(len=11) :: s3
  s1 = "Hello"
  s2 = " World"
  s3 = s1 // s2
  s2 = s1(2:4)
  print *, s3
  print *, s1(2:4)
end program character_ops

! CHECK: Hello World
! CHECK: ell

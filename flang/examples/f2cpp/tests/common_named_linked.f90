! RUN: echo '#include "f2cpp_rt.h"' > %t.named.main.cpp
! RUN: echo '#include "%t.named.main.h"' >> %t.named.main.cpp
! RUN: echo '#include "%t.named.helper.h"' >> %t.named.main.cpp
! RUN: echo '#include "%t.named.main.local.h"' >> %t.named.main.cpp
! RUN: echo '' >> %t.named.main.cpp
! RUN: %flang_fc1 -load %llvmshlibdir/flang2cpp%pluginext -plugin f2cpp -mllvm -declarations="%t.named.main.h"  -mllvm -local-declarations="%t.named.main.local.h" -mllvm -common-blocks="%t.named.main.common.c" %s >> %t.named.main.cpp

! RUN: echo '#include "f2cpp_rt.h"' > %t.named.helper.cpp
! RUN: echo '#include "%t.named.main.h"' >> %t.named.helper.cpp
! RUN: echo '#include "%t.named.helper.h"' >> %t.named.helper.cpp
! RUN: echo '#include "%t.named.helper.local.h"' >> %t.named.helper.cpp
! RUN: echo '' >> %t.named.helper.cpp
! RUN: %flang_fc1 -load %llvmshlibdir/flang2cpp%pluginext -plugin f2cpp -mllvm -declarations="%t.named.helper.h" -mllvm -local-declarations="%t.named.helper.local.h" -mllvm -common-blocks="%t.named.helper.common.c" %S/Inputs/common_named_linked_helper.f90 >> %t.named.helper.cpp

! RUN: %clang %t.named.main.common.c -I%S/../include -c -o %t.named.main.common.o
! RUN: %clang %t.named.helper.common.c -I%S/../include -c -o %t.named.helper.common.o

! RUN: %clangxx -c %t.named.main.cpp -I%S/../include -o %t.named.main.o
! RUN: %clangxx -c %t.named.helper.cpp -I%S/../include -o %t.named.helper.o

! RUN: %clangxx %t.named.main.o %t.named.helper.o %t.named.main.common.o %t.named.helper.common.o -L%llvmshlibdir -lf2cpp_rt -lflang_rt.runtime -o %t
! RUN: %t | FileCheck %s

! Test Named Common Block Linking
program common_named_linked
  integer :: x
  common /named_blk/ x
  x = 0
  call init_named_common()
  print *, "Named:", x
end program common_named_linked

! CHECK: Named: 30

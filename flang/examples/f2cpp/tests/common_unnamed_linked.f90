! RUN: echo '#include "f2cpp_rt.h"' > %t.main.cpp
! RUN: echo '#include "%{t:stem}.tmp.main.h"' >> %t.main.cpp
! RUN: echo '#include "%t.helper.h"' >> %t.main.cpp
! RUN: echo '#include "%t.main.local.h"' >> %t.main.cpp
! RUN: echo '' >> %t.main.cpp
! RUN: %flang_fc1 -load %llvmshlibdir/flang2cpp%pluginext -plugin f2cpp -mllvm -declarations="%t.main.h" -mllvm -local-declarations="%t.main.local.h" -mllvm -common-blocks="%t.main.common.c" %s >> %t.main.cpp

! RUN: echo '#include "f2cpp_rt.h"' > %t.helper.cpp
! RUN: echo '#include "%{t:stem}.tmp.helper.h"' >> %t.helper.cpp
! RUN: echo '#include "%t.main.h"' >> %t.helper.cpp
! RUN: echo '#include "%t.helper.local.h"' >> %t.helper.cpp
! RUN: echo '' >> %t.helper.cpp
! RUN: %flang_fc1 -load %llvmshlibdir/flang2cpp%pluginext -plugin f2cpp -mllvm -declarations="%t.helper.h" -mllvm -local-declarations="%t.helper.local.h" -mllvm -common-blocks="%t.helper.common.c" %S/Inputs/common_unnamed_linked_helper.f90 >> %t.helper.cpp

! RUN: %clang %t.main.common.c -I%S/../include -c -o %t.main.common.o
! RUN: %clang %t.helper.common.c -I%S/../include -c -o %t.helper.common.o

! RUN: %clangxx -c %t.main.cpp -I%S/../include -o %t.main.o
! RUN: %clangxx -c %t.helper.cpp -I%S/../include -o %t.helper.o

! RUN: %clangxx %t.main.o %t.helper.o %t.main.common.o %t.helper.common.o -L%llvmshlibdir -lf2cpp_rt -lflang_rt.runtime -o %t
! RUN: %t | FileCheck %s

! Test Unnamed Common Block Linking
program common_unnamed_linked
  integer :: a
  common // a
  a = 0
  call init_unnamed_common()
  print *, "Attributes:", a
end program common_unnamed_linked

! CHECK: Attributes: 10

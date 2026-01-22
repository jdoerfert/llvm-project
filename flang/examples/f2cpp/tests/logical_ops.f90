! RUN: echo '#include "f2cpp_rt.h"' > %t.cpp
! RUN: echo '' >> %t.cpp
! RUN: %flang_fc1 -load %llvmshlibdir/flang2cpp%pluginext -plugin f2cpp %s >> %t.cpp
! RUN: %clangxx %t.cpp -I%S/../include -L%llvmshlibdir -lf2cpp_rt -lflang_rt.runtime -o %t
! RUN: %t | FileCheck %s

! Test logical operations
program logical_ops
  logical :: t, f
  t = .true.
  f = .false.

  if (t .and. t) print *, "T and T"
  if (t .or. f) print *, "T or F"
  if (.not. f) print *, "not F"
  if (t .eqv. t) print *, "T eqv T"
  if (t .neqv. f) print *, "T neqv F"
end program logical_ops

! CHECK: T and T
! CHECK: T or F
! CHECK: not F
! CHECK: T eqv T
! CHECK: T neqv F

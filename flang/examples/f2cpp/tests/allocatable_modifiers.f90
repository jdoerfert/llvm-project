! RUN: echo '#include "f2cpp_rt.h"' > %t.cpp
! RUN: echo '' >> %t.cpp
! RUN: %flang_fc1 -load %llvmshlibdir/flang2cpp%pluginext -plugin f2cpp %s >> %t.cpp
! RUN: %clangxx %t.cpp -I%S/../include -L%llvmshlibdir -lf2cpp_rt -lflang_rt.runtime -o %t
! RUN: %t | FileCheck %s

! Test Allocatable Modifiers
module alloc_mod
  integer, allocatable, save :: saved_arr(:)
contains
  subroutine init_save()
    if (.not. allocated(saved_arr)) then
      allocate(saved_arr(2))
      saved_arr = 100
    else
      saved_arr = saved_arr + 1
    end if
    print *, "Saved:", saved_arr(1)
  end subroutine
end module

program allocatable_modifiers
  use alloc_mod
  integer, allocatable, target :: targ(:)
  integer, pointer :: p(:)
  integer, allocatable :: src(:), dst(:)

  ! Test Target
  allocate(targ(3))
  targ = 5
  p => targ
  p(2) = 9
  print *, "Target:", targ(1), targ(2), targ(3)
  deallocate(targ)

  ! Test Move_Alloc
  allocate(src(2))
  src = 42
  call move_alloc(src, dst)
  if (.not. allocated(src)) print *, "Source deallocated"
  if (allocated(dst)) print *, "Dest allocated:", dst(1)

  ! Test Save
  call init_save()
  call init_save()
end program allocatable_modifiers

! CHECK: Target: 5 9 5
! CHECK: Source deallocated
! CHECK: Dest allocated: 42
! CHECK: Saved: 100
! CHECK: Saved: 101

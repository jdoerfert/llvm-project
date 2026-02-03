! RUN: echo '#include "f2cpp_rt.h"' > %t.cpp
! RUN: echo '' >> %t.cpp
! RUN: %flang_fc1 -load %llvmshlibdir/flang2cpp%pluginext -plugin f2cpp %s >> %t.cpp
! RUN: %clangxx %t.cpp -I%S/../include -L%llvmshlibdir -lf2cpp_rt -lflang_rt.runtime -o %t
! RUN: %t | FileCheck %s

! Test Multidimensional Allocatable Arrays
program allocatable_multi_dim
  integer, allocatable :: mat(:,:), cube(:,:,:)
  integer :: i, j, k

  ! 2D Array
  allocate(mat(2, 3))
  do i = 1, 2
    do j = 1, 3
      mat(i, j) = i * 10 + j
    end do
  end do
  
  print *, "2D Mat:"
  do i = 1, 2
    print *, (mat(i, j), j=1, 3)
  end do
  deallocate(mat)

  ! 3D Array
  allocate(cube(2, 2, 2))
  cube = 0 ! Broadcast assignment
  cube(1, 1, 1) = 1
  cube(2, 2, 2) = 8
  
  print *, "3D Cube corner:", cube(1, 1, 1), cube(2, 2, 2)
  deallocate(cube)
end program allocatable_multi_dim

! CHECK: 2D Mat:
! CHECK: 11 12 13
! CHECK: 21 22 23
! CHECK: 3D Cube corner: 1 8

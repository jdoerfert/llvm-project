subroutine init_named_common()
  integer :: x, y
  common /named_blk/ x, y
  x = 30
  y = 40
end subroutine init_named_common

subroutine init_unnamed_common()
  integer :: a, b
  common // a, b
  a = 10
  b = 20
end subroutine init_unnamed_common

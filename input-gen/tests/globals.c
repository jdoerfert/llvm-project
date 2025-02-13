int X = 32;
__attribute__((weak)) int W;
struct {int a, b, c, d;} D;

__attribute__((inputgen_entry))
int foo() {
  X += W;
  return D.a;
}

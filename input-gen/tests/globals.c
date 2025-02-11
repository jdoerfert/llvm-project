int X = 32;
__attribute__((weak)) int W;
struct {int a, b, c, d;} D;

int foo() {
  X += W;
  return D.a;
}

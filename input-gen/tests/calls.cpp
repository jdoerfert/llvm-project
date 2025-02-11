#include <unistd.h>

void mysleep(int N) {
  sleep(N);
}
void bar(void (*fp)()) {
  mysleep(1);
  fp();
}
void foo() {
  mysleep(2);
}
int main() {
  mysleep(3);
  auto fp = &foo;
  bar(fp);
  sleep(1);
  foo();
}


#include <stdio.h>
#include <string.h>

void m(char *s) {
  if (!memcmp(s, "foo", 4)) {
    printf("Found foo\n");
  }
  if (!memcmp(s, "bar", 4)) {
    printf("Found bar\n");
  }
  printf("Really (%p): %s\n", s, s);
}

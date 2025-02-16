
#include <stdio.h>
#include <string.h>

__attribute__((inputgen_entry))
void m(char *s) {
  int i;
  if (!memcmp(s, "foo", 3)) {
    printf("Found foo at offset 0\n");
  }
  if (!memcmp(s, "bar", 3)) {
    printf("Found bar at offset 0\n");
  }
  if (!memcmp(s + 3, "foo", 4)) {
    printf("Found foo at offset 3\n");
  }
  if (!memcmp(s + 3, "bar", 4)) {
    printf("Found bar at offset 3\n");
  }
  if (!memcmp(s, "foobar", 7)) {
    printf("Found foobar at offset 0\n");
  }
  if (!memcmp(s, "barfoo", 7)) {
    printf("Found barfoo at offset 0\n");
  }
  printf("Final content (%p): '", (void*)s);
  for (i = 0; i < 7; ++i) {
    if (s[i])
      printf("%c", s[i]);
    else
      printf("_");
  }
  printf("'\n");
}

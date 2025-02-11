
char **A;

int main(int argc, char ** argv) {
  int s = 0;
  char *B = *A;
  for (int i = 0; i < argc; ++i)
    s+= B[i];
  return s;
}

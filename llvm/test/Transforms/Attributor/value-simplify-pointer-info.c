
static int rwtr = 3;
static int rwtn = 3;
static int rwtne = 3;
static int rwtnwc = 0;

void unknown(void);
void unknown_use(int *);

int read_write_test_recurse(void) {
  int r = rwtr;

  rwtr = 1;
  rwtr = 5;
  unknown();
  r += rwtr;
  rwtr = 1;

  rwtr = 7;
  unknown();
  r += rwtr;
  rwtr = 1;

  // may return 15 or something else.
  return r;
}

int read_write_test_norecurse(void) {
  int r = rwtn;

  rwtn = 1;
  rwtn = 5;
  unknown();
  r += rwtn;
  rwtn = 1;

  rwtn = 7;
  unknown();
  r += rwtn;
  rwtn = 1;

  // should return 15.
  return r;
}

int read_write_test_norecurse_esacpe(void) {
  int r = rwtne;

  rwtne = 1;
  rwtne = 5;
  unknown_use(&rwtne);
  r += rwtne;

  rwtne = 1;
  rwtne = 7;
  unknown_use(&rwtne);
  r += rwtne;
  rwtne = 1;

  // may return 15 or something else.
  return r;
}

static int read_write_test_norecurse_with_caller(void) {
  int r = rwtnwc;

  rwtnwc = 1;
  rwtnwc = 5;
  unknown();
  r += rwtnwc;

  rwtnwc = 1;
  rwtnwc = 7;
  unknown();
  r += rwtnwc;
  rwtnwc = 1;

  // should return 15.
  return r;
}
int read_write_test_norecurse_caller(void) {
  rwtnwc = 1;
  rwtnwc = 3;
  int r = read_write_test_norecurse_with_caller();
  rwtnwc = 1;
  // should return 15.
  return r;
}

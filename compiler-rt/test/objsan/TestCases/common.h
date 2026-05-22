#ifndef COMMON_H_
#define COMMON_H_

#include "objsan_interface.h"

#define OBJSAN_TEST_MAIN(test_name)                                            \
  int test_name(int argc, char **argv);                                        \
                                                                               \
  int main(int argc, char **argv) {                                            \
    __objsan_rt_init();                                                        \
    int result = test_name(argc, argv);                                        \
    __objsan_rt_deinit();                                                      \
    return result;                                                             \
  }

#endif // COMMON_H_

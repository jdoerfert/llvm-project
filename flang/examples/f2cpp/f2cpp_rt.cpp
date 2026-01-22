#include "f2cpp_rt.h"
#include <cstring>
#include <cstdio>

int main(int argc, const char **argv, const char **envp) {
  _FortranAProgramStart(argc, argv, envp, nullptr);
  _QQmain();
  _FortranAProgramEndStatement();
  return 0;
}

namespace flc {

void print_impl(void *Handle, int32_t First) {
  _FortranAioOutputInteger32(Handle, First);
}

void print_impl(void *Handle, int64_t First) {
  _FortranAioOutputInteger64(Handle, First);
}

void print_impl(void *Handle, float First) {
  _FortranAioOutputReal32(Handle, First);
}

void print_impl(void *Handle, double First) {
  _FortranAioOutputReal64(Handle, First);
}

void print_impl(void *Handle, char *First) {
  _FortranAioOutputAscii(Handle, First, std::strlen(First));
}

void print_impl(void *Handle, const char *First) {
  _FortranAioOutputAscii(Handle, First, std::strlen(First));
}

void print_impl(void *Handle, std::string &First) {
  _FortranAioOutputAscii(Handle, First.c_str(), First.size());
}

} // namespace flc

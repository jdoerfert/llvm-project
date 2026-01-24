//===- flcpp.h --------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
//===----------------------------------------------------------------------===//

#ifndef F2CPP_RT_H
#define F2CPP_RT_H

#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <type_traits>

#include "f2cpp_rt_macros.h"
#include "f2cpp_rt_types.h"

extern "C" {
void _FortranAProgramStart(int, const char **, const char **, void *);
void _FortranAProgramEndStatement();
void *_FortranAioBeginExternalListInput(int32_t, const char *, int32_t);
void *_FortranAioBeginExternalListOutput(int32_t, const char *, int32_t);
bool _FortranAioInputAscii(void *, const char *, int64_t);
bool _FortranAioOutputAscii(void *, const char *, int64_t);
bool _FortranAioOutputDescriptor(void *, void *);
bool _FortranAioOutputReal32(void *, float);
bool _FortranAioOutputReal64(void *, double);
bool _FortranAioOutputInteger32(void *, int32_t);
bool _FortranAioOutputInteger64(void *, int64_t);
int32_t _FortranAioEndIoStatement(void *);

void _QQmain();
}

template <typename T, size_t N> struct StaticArray {
  T Data[N];

  StaticArray() = default;

  StaticArray(T (&Arr)[N]) { memcpy(Data, Arr, N); }

  StaticArray<T, N>(std::initializer_list<T> List) {
    std::copy(List.begin(), List.end(), Data);
  }

  // Overload the assignment operator
  StaticArray<T, N> &operator=(const StaticArray<T, N> &other) {
    if (this != &other) { // Self-assignment check
      std::copy(other.Data, other.Data + N, Data);
    }
    return *this;
  }
  StaticArray<T, N> &operator=(const T Other[N]) {
    std::copy(&Other[0], &Other[0] + N, Data);
    return *this;
  }

  T *data() { return Data; }
  const T *data() const { return Data; }

  operator T *() { return &Data[0]; }
  operator const T *() const { return &Data[0]; }

  template <typename IntT> T &operator()(const IntT &I) {
    return (*this)[I - 1];
  }

  StaticArray<T, N> operator+(StaticArray<T, N> &Other) {
    StaticArray<T, N> R;
    for (int I = 0; I < N; ++I)
      R.Data[I] = Data[I] + Other[I];
    return R;
  }

  T &operator[](size_t index) { return Data[index]; }

  const T &operator[](size_t index) const { return Data[index]; }
};

template <size_t N> struct CHARACTER : public StaticArray<char, N> {
  CHARACTER<N> &operator=(const std::string &S) {
    auto Size = S.size();
    if (Size >= N) {
      memcpy(this->data(), S.data(), N);
    } else {
      memcpy(this->data(), S.data(), Size);
      memset(this->data() + Size, ' ', N - Size);
    }
    return *this;
  }
};

///
///{
template <typename T> std::string CHAR(T Arg) { return std::to_string(Arg); }

template <typename T> int INT(T Arg) { return (int)Arg; }

template <size_t N> int LEN(CHARACTER<N> &) { return N; }
template <size_t N> int LEN(const char (&)[N]) { return N; }
inline int LEN(const std::string &S) { return S.size(); }

template <typename T, size_t N> std::string TRIM(T *Arr) {
  int I = N - 1;
  while (I >= 0 && Arr[I] == ' ')
    I--;
  ++I;
  std::string R;
  R.resize(I);
  std::memcpy(R.data(), Arr, sizeof(T) * I);
  return R;
}
template <typename T, size_t N> std::string TRIM(T (&Arr)[N]) {
  return TRIM<T, N>(&Arr[0]);
}
template <size_t N> std::string TRIM(CHARACTER<N> &Arr) {
  return TRIM<char, N>(&Arr[0]);
}

///}
///

namespace flc {

template <typename U> void print_impl(void *Handle, U First) {
  printf("TODO1 %p ", (void *)First);
}

template <typename T, size_t N>
void print_impl(void *Handle, StaticArray<T, N> &First) {
  struct D {
    void *base_addr;
    size_t elem_len;
    int32_t version;
    int8_t rank, type, attribute, extra;
    int64_t dim[3];
  } d;
  d.base_addr = First.data();
  d.elem_len = sizeof(T);
  d.version = 20240719;
  d.rank = 1;
  d.type = /* TODO real?= */ 27;
  d.attribute = 0;
  d.extra = 0;
  d.dim[0] = 1;
  d.dim[1] = N;
  d.dim[2] = sizeof(T);
  _FortranAioOutputDescriptor(Handle, &d);
}

template <size_t N> void print_impl(void *Handle, CHARACTER<N> &First) {
  _FortranAioOutputAscii(Handle, First, N);
}
template <typename U, typename... Ts>
void print_impl(void *Handle, U First, Ts... Args) {
  printf("TODO2 %p ", (void *)First);
  print_impl(Handle, Args...);
}

#define SIMPLE_PRINT_TYPES \
  X(float) \
  X(double) \
  X(int32_t) \
  X(int64_t) \
  X(char *) \
  X(const char *) \
  X(std::string &)

#define PRINT_IMPL_LEAF_DECL(TYPE) void print_impl(void *Handle, TYPE First);

#define PRINT_IMPL_NON_LEAF_DECL(TYPE) \
  template <typename... Ts> \
  void print_impl(void *Handle, TYPE First, Ts... Args);

#define PRINT_IMPL_NON_LEAF_DEF(TYPE) \
  template <typename... Ts> \
  void print_impl(void *Handle, TYPE First, Ts... Args) { \
    print_impl(Handle, First); \
    print_impl(Handle, Args...); \
  }

#define X(T) PRINT_IMPL_LEAF_DECL(T)
SIMPLE_PRINT_TYPES
#undef X

#define X(T) PRINT_IMPL_NON_LEAF_DECL(T)
SIMPLE_PRINT_TYPES
#undef X

#define X(T) PRINT_IMPL_NON_LEAF_DEF(T)
SIMPLE_PRINT_TYPES
#undef X

template <typename... Ts>
void print(const char *UsrFormat, const char *FILE, int LINE, Ts... Args) {
  if (std::string(UsrFormat) != "*") {
    printf("TODO: Implement print format %s\n", UsrFormat);
    return;
  }
  void *Handle = _FortranAioBeginExternalListOutput(6, FILE, LINE);
  print_impl(Handle, Args...);
  _FortranAioEndIoStatement(Handle);
}

// READ
//

template <typename T, size_t N> constexpr size_t arraySize(T (&)[N]) {
  return N;
}

template <typename T, size_t N>
constexpr size_t arraySize(StaticArray<T, N> &) {
  return N;
}

template <typename T> void read_impl(void *Handle, T *First, size_t Size) {
  _FortranAioInputAscii(Handle, First, Size * sizeof(T));
}

template <typename T>
void read(
    const char *UsrFormat, const char *FILE, int LINE, T *Arg, size_t Size) {
  if (std::string(UsrFormat) != "*") {
    printf("TODO: Implement read format %s\n", UsrFormat);
    return;
  }
  void *Handle = _FortranAioBeginExternalListInput(5, FILE, LINE);
  read_impl(Handle, Arg, Size);
  _FortranAioEndIoStatement(Handle);
}
}; // namespace flc

#endif
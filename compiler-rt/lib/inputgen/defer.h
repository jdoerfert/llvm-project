#ifndef DEFER_H_
#define DEFER_H_

#include "logging.h"

#include <cassert>
#include <cstdio>
#include <utility>

namespace __ig {
template <class T> class DeferLocalConstruction {
public:
  DeferLocalConstruction() {}
  ~DeferLocalConstruction() { V.~T(); }
  template <typename... ArgsTy> void init(ArgsTy &&...Args) {
    new (&V) T(std::forward<ArgsTy>(Args)...);
  }
  T *operator->() { return &V; }
  T &operator*() { return V; }

private:
  union {
    T V;
  };
};

template <class T, bool &Init> class DeferGlobalConstruction {
public:
  DeferGlobalConstruction() {}

  ~DeferGlobalConstruction() {
    if (Init) {
      INPUTGEN_DEBUG(fputs("DEFERRED CONSTRUCT\n", stderr));
      V.~T();
    }
  }

  template <typename... ArgsTy> void init(ArgsTy &&...Args) {
    if (!Init) {
      // We cannot use std::cerr here because we may have not run global
      // constructors yet.
      INPUTGEN_DEBUG(fputs("DEFERRED DESTRUCT\n", stderr));
      new (&V) T(std::forward<ArgsTy>(Args)...);
      Init = true;
    }
  }

  T *operator->() {
    assert(Init);
    return &V;
  }

  T &operator*() {
    assert(Init);
    return V;
  }

  bool isConstructed() { return Init; }

private:
  union {
    T V;
  };
};
} // namespace __ig

#endif // DEFER_H_

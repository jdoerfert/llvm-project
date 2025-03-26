#include "include/obj_encoding.h"

#include <new>

namespace __objsan {

template <class T> struct LazyConstruction {
  static bool Init;
  LazyConstruction() {}
  ~LazyConstruction() {
    if (Init)
      V.~T();
  }
  template <typename... ArgsTy> void construct(ArgsTy &&...Args) {
    if (!Init) {
      new (&V) T(std::forward<ArgsTy>(Args)...);
      Init = true;
    }
  }
  union {
    T V;
  };
};

static LazyConstruction<SmallObjectsTy> LazySmallObjects;
static LazyConstruction<LargeObjectsTy> LazyLargeObjects;
template <> bool LazyConstruction<SmallObjectsTy>::Init = false;
template <> bool LazyConstruction<LargeObjectsTy>::Init = false;

__attribute__((visibility("default"))) SmallObjectsTy &SmallObjects =
    LazySmallObjects.V;
__attribute__((visibility("default"))) LargeObjectsTy &LargeObjects =
    LazyLargeObjects.V;

SmallObjectsTy &getOrConstructSmallObjects() {
  LazySmallObjects.construct();
  return SmallObjects;
}
LargeObjectsTy &getOrConstructLargeObjects() {
  LazyLargeObjects.construct();
  return LargeObjects;
}

__attribute((constructor)) void initialize() {
  // Ensure the globals are constructed before the program begins. If it is
  // multithreaded, we do not want multiple threads to initialize the objects.
  getOrConstructSmallObjects();
  getOrConstructLargeObjects();
}

} // namespace __objsan

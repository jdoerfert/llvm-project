#include "include/obj_encoding.h"

namespace __objsan {

template <typename T> class ObjectDeallocator {
  T *&ObjectPtr;

public:
  ObjectDeallocator(T *&Ptr) : ObjectPtr(Ptr) {}
  ~ObjectDeallocator() {
    if (ObjectPtr == nullptr)
      delete ObjectPtr;
  }
};

__attribute__((visibility("default"))) SmallObjectsTy *SmallObjects = nullptr;
__attribute__((visibility("default"))) LargeObjectsTy *LargeObjects = nullptr;

// These will ensure the objects are deallocated when the program ends.
ObjectDeallocator<SmallObjectsTy> SODeallocator(SmallObjects);
ObjectDeallocator<LargeObjectsTy> LODeallocator(LargeObjects);

__attribute((constructor)) void initialize() {
  // Ensure the globals are constructed before the program begins. If it is
  // multithreaded, we do not want multiple threads to initialize the objects.
  getSmallObjects();
  getLargeObjects();
}

} // namespace __objsan

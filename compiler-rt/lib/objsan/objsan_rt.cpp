#include "include/obj_encoding.h"

namespace __objsan {
__attribute__((visibility("default"))) SmallObjectsTy SmallObjects;

__attribute__((visibility("default"))) LargeObjectsTy LargeObjects;

__attribute__((visibility("default"))) FixedObjectsTy FixedObjects;
} // namespace __objsan

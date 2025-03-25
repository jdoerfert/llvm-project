#include "include/obj_encoding.h"

namespace __objsan {
__attribute__((visibility("default"))) SmallObjectsTy SmallObjects;

__attribute__((visibility("default"))) LargeObjectsTy LargeObjects;
} // namespace __objsan

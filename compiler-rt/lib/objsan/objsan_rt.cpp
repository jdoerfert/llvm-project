#include "include/obj_encoding.h"

namespace __objsan {

__attribute__((visibility("default"))) SmallObjectsTy SmallObjects;
__attribute__((visibility("default"))) LargeObjectsTy LargeObjects;

#ifdef STATS
#ifndef __OBJSAN_DEVICE__
__attribute__((visibility("default"))) StatsTy SLoads("loads");
__attribute__((visibility("default"))) StatsTy SStores("stores");
__attribute__((visibility("default"))) StatsTy SRange("range");
__attribute__((visibility("default"))) StatsTy SLoopR("loopr");
#endif
#endif
} // namespace __objsan

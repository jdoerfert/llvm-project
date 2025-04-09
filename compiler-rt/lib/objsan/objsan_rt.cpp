#include "include/obj_encoding.h"

extern "C" {

__attribute__((visibility("default"))) __objsan::SmallObjectsTy __objsan_SmallObjects;
__attribute__((visibility("default"))) __objsan::LargeObjectsTy __objsan_LargeObjects;

};

namespace __objsan {

#ifdef STATS
#ifndef __OBJSAN_DEVICE__
__attribute__((visibility("default"))) StatsTy SLoads("loads");
__attribute__((visibility("default"))) StatsTy SStores("stores");
__attribute__((visibility("default"))) StatsTy SRange("range");
__attribute__((visibility("default"))) StatsTy SLoopR("loopr");
#endif
#endif
} // namespace __objsan

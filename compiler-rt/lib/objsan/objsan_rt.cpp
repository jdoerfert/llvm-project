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

extern "C" {
using CtorFn = void (*)(void);
extern CtorFn __start___objsan_ctor;
extern CtorFn __stop___objsan_ctor;

__attribute__((constructor(1000))) void __objsan_ctor_init() {
  //  fprintf(stderr, "CTOR INIT  %p %p, %lu\n", &__start___objsan_ctor,
  //          &__stop___objsan_ctor,
  //          &__stop___objsan_ctor - &__start___objsan_ctor);
  for (CtorFn *Ctor = &__start___objsan_ctor, *E = &__stop___objsan_ctor;
       Ctor != E; ++Ctor)
    (*Ctor)();
}
}

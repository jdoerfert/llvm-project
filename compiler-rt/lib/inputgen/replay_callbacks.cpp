#include <cstdio>
#include <cstring>

#include "common.h"
#include "defer.h"
#include "global_manager.h"

#ifndef NDEBUG
#define PRINTF(...) fprintf(stderr, __VA_ARGS__)
#else
#define PRINTF(...)
#endif

namespace __ig {
extern bool GMInit;
extern DeferGlobalConstruction<GlobalManager, GMInit> GM;
} // namespace __ig

using namespace __ig;

extern "C" {

IG_API_ATTRS
void __ig_pre_module(char *module_name, char *name, int32_t id) {
  PRINTF("module pre -- module_name: %s, name: %s, id: %i\n", module_name, name,
         id);
  GM.init();
}

IG_API_ATTRS
void __ig_post_module(char *module_name, char *name, int32_t id) {
  PRINTF("module post -- module_name: %s, name: %s, id: %i\n", module_name,
         name, id);
}

IG_API_ATTRS
void __ig_pre_global_ind(char *address, char *name, char *initial_value_ptr,
                         int32_t initial_value_size, int8_t is_constant) {
  GM->addGlobal(address, name, initial_value_size, (bool)is_constant);
}

IG_API_ATTRS
void __ig_pre_global(char *address, char *name, int64_t initial_value,
                     int32_t initial_value_size, int8_t is_constant) {
  PRINTF("global pre -- address: %p, name: %s, initial_value: %lli, "
         "initial_value_size: %i, is_constant: %i\n",
         address, name, initial_value, initial_value_size, is_constant);
  return __ig_pre_global_ind(address, name, (char *)&initial_value,
                             initial_value_size, is_constant);
}

IG_API_ATTRS
void __ig_gen_value(void *pointer, int32_t value_size, int64_t alignment,
                    int32_t value_type_id) {
  PRINTF("load pre -- pointer: %p, value_size: %i, alignment: %lli, "
         "value_type_id: %i\n",
         pointer, value_size, alignment, value_type_id);
  memset(pointer, 0, value_size);
}
}

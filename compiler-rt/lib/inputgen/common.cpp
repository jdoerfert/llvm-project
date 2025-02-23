#include <cstdint>

#include "common.h"
#include "logging.h"

namespace __ig {
void printAvailableFunctions() {
  ERR("  Available functions:\n");
  for (uint32_t I = 0; I < __ig_num_entry_points; I++)
    ERR("    {}: {}\n", I, __ig_entry_point_names[I]);
}
void printNumAvailableFunctions() {
  ERR("  Num available functions: {}\n", __ig_num_entry_points);
}
} // namespace __ig

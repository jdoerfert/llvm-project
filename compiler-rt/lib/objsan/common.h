#ifndef COMMON_H
#define COMMON_H

#include <cstdint>

extern "C" char *__ig_entry_point_names[];
extern "C" uint32_t __ig_num_entry_points;

#define IG_API_ATTRS __attribute__((always_inline))

namespace __ig {

void printAvailableFunctions();
void printNumAvailableFunctions();

enum class ExitStatus : int {
  Success = 0,
  EntryNoOutOfBounds,
  NoInputs,
  WrongUsage,
};

} // namespace __ig

#endif // COMMON_H

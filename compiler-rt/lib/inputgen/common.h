#ifndef COMMON_H
#define COMMON_H

#include "logging.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <vector>

extern "C" void __ig_entry(uint32_t, void *);

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

// clang-format off
#define BYTE_TO_BINARY_PATTERN "%c%c%c%c %c%c%c%c "
#define BYTE_TO_BINARY(byte)  \
  ((byte) & 0x08 ? 'S' : '-'),\
  ((byte) & 0x04 ? 'R' : '-'),\
  ((byte) & 0x02 ? 'P' : '-'),\
  ((byte) & 0x01 ? 'I' : '-'),\
  ((byte) & 0x80 ? 'S' : '-'),\
  ((byte) & 0x40 ? 'R' : '-'),\
  ((byte) & 0x20 ? 'P' : '-'),\
  ((byte) & 0x10 ? 'I' : '-')
// clang-format on

inline void dumpMemoryBinary(char *Memory, size_t Size) {
  fprintf(stderr, "[%p] ", Memory);
  for (uint32_t I = 0; I < Size; ++I) {
    if (I % 16 == 0)
      fprintf(stderr, "\n[+%d]\t", I);
    fprintf(stderr, BYTE_TO_BINARY_PATTERN, BYTE_TO_BINARY(Memory[I]));
  }
  fputs("\n", stderr);
}

inline void dumpMemoryHex(char *Memory, size_t Size) {
  fprintf(stderr, "[%p] ", Memory);
  for (uint32_t I = 0; I < Size; ++I) {
    if (I % 16 == 0)
      fprintf(stderr, "\n[+%d]\t", I);
    fprintf(stderr, "%02hhX ", Memory[I]);
  }
  fputs("\n", stderr);
}

inline std::optional<int> getIntEnv(const char *Name) {
  if (char *C = getenv(Name)) {
    INPUTGEN_DEBUG(fprintf(stderr, "Got env option %s: %s\n", Name, C));
    return std::atoi(C);
  }
  return std::nullopt;
}

} // namespace __ig

#endif // COMMON_H

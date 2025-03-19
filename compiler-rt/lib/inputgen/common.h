#ifndef COMMON_H
#define COMMON_H

#include <cstdint>
#include <cstdio>
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

#define BYTE_TO_BINARY_PATTERN "%c%c%c%c%c%c%c%c "
#define BYTE_TO_BINARY(byte)                                                   \
  ((byte) & 0x80 ? '1' : '0'), ((byte) & 0x40 ? '1' : '0'),                    \
      ((byte) & 0x20 ? '1' : '0'), ((byte) & 0x10 ? '1' : '0'),                \
      ((byte) & 0x08 ? '1' : '0'), ((byte) & 0x04 ? '1' : '0'),                \
      ((byte) & 0x02 ? '1' : '0'), ((byte) & 0x01 ? '1' : '0')

inline void dumpMemoryBinary(char *Memory, size_t Size) {
  for (uint32_t I = 0; I < Size; ++I) {
    if (I % 16 == 0)
      fprintf(stderr, "(%p) ", (void *)(Memory + I));
    fprintf(stderr, BYTE_TO_BINARY_PATTERN, BYTE_TO_BINARY(Memory[I]));
  }
  fputs("\n", stderr);
}

inline void dumpMemoryHex(char *Memory, size_t Size) {
  for (uint32_t I = 0; I < Size; ++I) {
    if (I % 16 == 0)
      fprintf(stderr, "(%p) ", (void *)(Memory + I));
    fprintf(stderr, "%02hhX ", Memory[I]);
  }
  fputs("\n", stderr);
}

} // namespace __ig

#endif // COMMON_H

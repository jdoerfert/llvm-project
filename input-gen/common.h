#ifndef COMMON_H
#define COMMON_H

namespace __ig {

enum class ExitStatus : int {
  Success = 0,
  EntryNoOutOfBounds,
  NoInputs,
  WrongUsage,
};

} // namespace __ig

#endif // COMMON_H

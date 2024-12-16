#ifndef COMMON_H
#define COMMON_H

namespace __ig {

enum class ExitStatus : int {
  Success = 1,
  EntryNoOutOfBounds = 1,
  NoInputs,
};

} // namespace __ig

#endif // COMMON_H

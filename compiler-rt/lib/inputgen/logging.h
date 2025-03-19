#ifndef LOGGING_H
#define LOGGING_H

#include <format>
#include <iostream>

namespace __ig {

#ifndef NDEBUG
#define INPUTGEN_DEBUG(X)                                                      \
  do {                                                                         \
    X;                                                                         \
  } while (0)
#else
#define INPUTGEN_DEBUG(X)
#endif

template <typename... Args>
void INFO(const std::format_string<Args...> S, Args &&...As) {
  fputs(std::format(S, std::forward<Args>(As)...).c_str(), stderr);
}

template <typename... Args>
void ERR(const std::format_string<Args...> S, Args &&...As) {
  fputs(std::format(S, std::forward<Args>(As)...).c_str(), stderr);
}

template <typename... Args>
void VERBOSE(const std::format_string<Args...> S, Args &&...As) {
  INPUTGEN_DEBUG(
      fputs(std::format(S, std::forward<Args>(As)...).c_str(), stderr));
}

template <typename... Args>
void DEBUG(const std::format_string<Args...> S, Args &&...As) {
  INPUTGEN_DEBUG(
      fputs(std::format(S, std::forward<Args>(As)...).c_str(), stderr));
}

template <typename... Args>
void WARN(const std::format_string<Args...> S, Args &&...As) {
  INPUTGEN_DEBUG(
      fputs(std::format(S, std::forward<Args>(As)...).c_str(), stderr));
}

} // namespace __ig

#endif // LOGGING_H

#ifndef LOGGING_H
#define LOGGING_H

#include <format>
#include <iostream>

namespace __ig {

template <typename... Args>
void INFO(const std::format_string<Args...> S, Args &&...As) {
  std::cout << std::format(S, std::forward<Args>(As)...);
}

template <typename... Args>
void VERBOSE(const std::format_string<Args...> S, Args &&...As) {
#ifndef NDEBUG
  std::cout << std::format(S, std::forward<Args>(As)...);
#endif
}

template <typename... Args>
void DEBUG(const std::format_string<Args...> S, Args &&...As) {
#ifndef NDEBUG
  std::cerr << std::format(S, std::forward<Args>(As)...);
#endif
}

template <typename... Args>
void WARN(const std::format_string<Args...> S, Args &&...As) {
#ifndef NDEBUG
  std::cerr << std::format(S, std::forward<Args>(As)...);
#endif
}

template <typename... Args>
void ERR(const std::format_string<Args...> S, Args &&...As) {
  std::cerr << std::format(S, std::forward<Args>(As)...);
}

} // namespace __ig

#endif // LOGGING_H

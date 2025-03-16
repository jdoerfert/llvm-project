#ifndef GLOBAL_MANAGER_H_
#define GLOBAL_MANAGER_H_

#include <cstdint>
#include <utility>
#include <vector>

namespace __ig {
struct GlobalManager {
  ~GlobalManager() {}
  GlobalManager() {}
  template <typename... ArgsTy> void addGlobal(ArgsTy &&...Args) {
    GlobalTy Global{std::forward<ArgsTy>(Args)...};
    if (!Global.IsConstant)
      Globals.push_back(Global);
  }
  struct GlobalTy {
    char *Address;
    char *Name;
    int32_t Size;
    bool IsConstant;
  };
  std::vector<GlobalTy> Globals;
};
} // namespace __ig

#endif // GLOBAL_MANAGER_H_

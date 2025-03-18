#ifndef GLOBAL_MANAGER_H_
#define GLOBAL_MANAGER_H_

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace __ig {
struct GlobalManager {
  ~GlobalManager() {}
  GlobalManager() {}
  template <typename... ArgsTy> void addGlobal(ArgsTy &&...Args) {
    GlobalTy Global{std::forward<ArgsTy>(Args)...};
    if (!Global.Name)
      Global.Name = "";
    if (!Global.IsConstant)
      Globals.push_back(Global);
  }
  struct GlobalTy {
    char *Address;
    const char *Name;
    int32_t Size;
    bool IsConstant;

    bool operator<(const GlobalTy &Other) const {
      return strcmp(Name, Other.Name) < 0;
    }
  };
  std::vector<GlobalTy> Globals;

  void sort() { std::sort(Globals.begin(), Globals.end()); }
};
} // namespace __ig

#endif // GLOBAL_MANAGER_H_

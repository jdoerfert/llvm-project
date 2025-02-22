#ifndef VM_VALUES_H
#define VM_VALUES_H

#include <cassert>
#include <cstdint>
#include <deque>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace __ig {

struct FreeValueManager;
struct FreeValueInfo {
  const uint32_t TypeId;
  const uint32_t Size;
  char *VPtr;
  char *MPtr;
  char *VCmpPtr = nullptr;
  char *MCmpPtr = nullptr;
  size_t CmpSize = 0;

  FreeValueInfo(uint32_t TypeId, uint32_t Size, char *VPtr);
  FreeValueInfo(uint32_t TypeId, uint32_t Size, char *VPtr, char *VCmpPtr,
                size_t CmpSize);

  template <typename Ty> void setValues(Ty *V, uint32_t NV) {
    Values = (char *)V;
    NumValues = NV;
    ValueSize = sizeof(Ty);
  }

  uint32_t getNumValues(FreeValueManager &FVM) const;
  char *getValuePtr(FreeValueManager &FVM) const;
  size_t getValueSize(FreeValueManager &FVM) const;
  size_t getWrittenSize(FreeValueManager &FVM) const;

  bool isInitialized() const { return UsesIndirectValues || !!Values; }
  bool isFixed();

  char *write(FreeValueManager &FVM);
  void markInitialized(FreeValueManager &FVM, char *VP, char *VPC);

  uint32_t Idx = 0;
  bool UsesIndirectValues = false;

private:
  char *Values = nullptr;
  uint32_t ValueSize = 0;
  uint32_t NumValues = 0;
  bool IsFixed = false;

  bool isMemcmp() const { return VCmpPtr; }
};

struct BranchConditionInfo {
  std::vector<FreeValueInfo> FreeValueInfos;
  uint32_t No;
  using FnTy = char (*)(void *);
  FnTy Fn;
  char *ArgMemPtr;
  bool IsFixed = false;
};

struct FreeValueDecisionTy {};

static char *PtrValues[] = {0};
static int32_t I32Values[] = {
    0,   1,   2,   3,    4,    8,   12,  16, 22, 24, 26, 32, 64,
    128, 256, 512, 1024, -100, -64, -32, -8, -4, -3, -2, -1,
};
static uint32_t NumI32Values = sizeof(I32Values) / sizeof(I32Values[0]);

struct FreeValueManager {

  FreeValueManager() {
    StringCache.emplace_back(256, '\0');
    StringCacheSet.insert(StringCache.back());
  }

  using FreeValueVecTy = std::vector<FreeValueInfo *>;
  using BCIVecTy = std::vector<BranchConditionInfo *>;
  using BCISetTy = std::set<BranchConditionInfo *>;

  void reset();

  std::vector<BranchConditionInfo *> BranchConditionMap;
  std::map<char *, BCIVecTy> BranchConditions;
  std::set<std::string_view> StringCacheSet;
  std::deque<std::string> StringCache;

  BCIVecTy *lookupBCIVec(char *VPtr) {
    auto BCIt = BranchConditions.find(VPtr);
    if (BCIt == BranchConditions.end())
      return nullptr;
    return &BCIt->second;
  }

  bool isFreeValue(BranchConditionInfo &BCI, FreeValueInfo &FVI);
  void checkBranchConditions(char *VPtr, char *VPBP, char *VCmpPtr,
                             char *VCPBP);

  bool workOn(FreeValueVecTy &FVVec, BCIVecTy &BCIs);
  bool modifyAndEvaluate(FreeValueInfo &FVI, BCIVecTy &BCIs);

  bool evaluate(BranchConditionInfo &BCI, bool B = false);
};

} // namespace __ig

#endif

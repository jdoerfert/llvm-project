#ifndef VM_VALUES_H
#define VM_VALUES_H

#include <cassert>
#include <cstdint>
#include <map>
#include <set>

namespace __ig {

struct FreeValueInfo {
  const uint32_t Offset;
  const uint32_t TypeId;
  const uint32_t Size;
  char *const VPtr;
  char *const VCmpPtr = nullptr;
  const size_t CmpSize = 0;

  FreeValueInfo(uint32_t Offset, uint32_t TypeId, uint32_t Size, char *VPtr)
      : Offset(Offset), TypeId(TypeId), Size(Size), VPtr(VPtr) {}
  FreeValueInfo(uint32_t Offset, uint32_t TypeId, uint32_t Size, char *VPtr,
                char *VCmpPtr, size_t CmpSize)
      : Offset(Offset), TypeId(TypeId), Size(Size), VPtr(VPtr),
        VCmpPtr(VCmpPtr), CmpSize(CmpSize) {};

  bool isMemcmp() const { return VCmpPtr; }

  size_t getManifestSize() const { return isMemcmp() ? CmpSize : Size; }
};

struct BranchConditionInfo {
  std::vector<FreeValueInfo> FreeValueInfos;
  uint32_t No;
  using FnTy = char (*)(void *);
  FnTy Fn;
  char *ArgMemPtr;
};

struct FreeValueDecisionTy {
  char *Values = nullptr;
  uint32_t Idx = 0;
  uint32_t ValueSize = 0;
  uint32_t NumValues = 0;
  bool IsFixed = false;
  bool IsSwappedMemcmp = false;

  template <typename Ty> void setValues(Ty *V, uint32_t NV) {
    assert(!Values);
    Values = (char *)V;
    NumValues = NV;
    ValueSize = sizeof(Ty);
  }

  char *getDstVPtr(FreeValueInfo &FVI) {
    return IsSwappedMemcmp ? FVI.VCmpPtr : FVI.VPtr;
  }
  char *getSrcVPtr(FreeValueInfo &FVI) {
    if (FVI.isMemcmp())
      return IsSwappedMemcmp ? FVI.VPtr : FVI.VCmpPtr;
    return getValuePtr();
  }
  char *getSrcMPtr(FreeValueInfo &FVI);
  char *getValuePtr() const { return &Values[Idx * ValueSize]; }
  bool isInitialized() const { return Values; }
  bool isFixed(FreeValueInfo &FVI, bool &Consistent);
  bool isConsistent(FreeValueInfo &FVI, char *Ptr1, char *Ptr2);
  bool isConsistent(FreeValueInfo &FVI, FreeValueDecisionTy &OtherFVD, FreeValueInfo &OtherFVI);
  bool willManifest(FreeValueInfo &FVI) const {
    return !FVI.isMemcmp() || Idx == 1;
  }

  char *write(char *DstMPtr, FreeValueInfo &FVI, bool IsManifest = false);
  char *write(BranchConditionInfo &BCI, FreeValueInfo &FVI,
              bool IsManifest = false) {
    return write(BCI.ArgMemPtr + FVI.Offset, FVI, IsManifest);
  }
  void manifest(BranchConditionInfo &BCI, FreeValueInfo &FVI);
};

static int32_t MemcmpValues[] = {-1, 0, 1};
static uint32_t NumMemcmpValues = 3;

static int32_t I32Values[] = {-100, -64, -32, -8,  -4,  -3,  -2,  -1, 0,
                              1,    2,   3,   4,   8,   12,  16,  22, 24,
                              26,   32,  64,  128, 256, 512, 1024};
static uint32_t NumI32Values = sizeof(I32Values) / sizeof(I32Values[0]);

struct FreeValueManager {

  using BCIVecTy = std::vector<BranchConditionInfo *>;
  using BCISetTy = std::set<BranchConditionInfo *>;

  void reset();

  using WriteMapTy =
      std::map<char *,
               std::vector<std::tuple<FreeValueDecisionTy *, FreeValueInfo *,
                                      BranchConditionInfo *>>>;
  using DecisionInfluenceMapTy =
      std::map<FreeValueDecisionTy *,
               std::vector<std::pair<BranchConditionInfo *, FreeValueInfo *>>>;
  using FreeDecisionMapTy =
      std::map<BranchConditionInfo *,
               std::vector<std::pair<FreeValueInfo *, FreeValueDecisionTy *>>>;

  std::map<std::pair<char *, char *>, FreeValueDecisionTy *> FreeValueDecisions;

  std::map<char *, BCIVecTy> BranchConditions;

  BCIVecTy *lookupBCIVec(char *VPtr) {
    auto BCIt = BranchConditions.find(VPtr);
    if (BCIt == BranchConditions.end())
      return nullptr;
    return &BCIt->second;
  }

  FreeValueDecisionTy *lookupFreeValueDecision(FreeValueInfo &FVI) {
    auto It = FreeValueDecisions.find({FVI.VPtr, FVI.VCmpPtr});
    return It == FreeValueDecisions.end() ? nullptr : It->second;
  }
  FreeValueDecisionTy &getFreeValueDecision(FreeValueInfo &FVI) {
    assert(FreeValueDecisions.count({FVI.VPtr, FVI.VCmpPtr}));
    return *FreeValueDecisions[{FVI.VPtr, FVI.VCmpPtr}];
  }
  FreeValueDecisionTy &getFreeValueDecision(char *VPtr, char *VCmpPtr) {
    assert(FreeValueDecisions.count({VPtr, VCmpPtr}));
    return *FreeValueDecisions[{VPtr, VCmpPtr}];
  }
  FreeValueDecisionTy &getOrCreateFreeValueDecision(FreeValueInfo &FVI) {
    auto *&FVD = FreeValueDecisions[{FVI.VPtr, FVI.VCmpPtr}];
    if (!FVD)
      FVD = new FreeValueDecisionTy;
    return *FVD;
  }

  FreeValueDecisionTy *identifyFreeValueDecision(BranchConditionInfo &BCI,
                                                 FreeValueInfo &FVI,
                                                 bool &Consistent);
  void checkBranchConditions(char *VPtr, char *VCmpPtr);

  bool workOn(BranchConditionInfo &BCI, FreeDecisionMapTy &FreeDecisionMap,
              DecisionInfluenceMapTy &DecisionInfluenceMap,
              WriteMapTy &WriteMap, BCISetTy &BadBSIs);
  bool modifyAndEvaluate(FreeValueDecisionTy &FVD, FreeValueInfo &FVI,
                         DecisionInfluenceMapTy &DecisionInfluenceMap,
                         WriteMapTy &WriteMap, BCISetTy &BadBSIs);
  bool checkConsistency(FreeValueDecisionTy &FVD, FreeValueInfo &FVI,
                        WriteMapTy &WriteMap, BCISetTy &BadBSIs);

  bool evaluate(BranchConditionInfo &BCI);
};

} // namespace __ig

#endif

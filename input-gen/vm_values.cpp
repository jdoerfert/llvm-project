
#include "vm_values.h"
#include "logging.h"
#include "vm_obj.h"

#include <algorithm>
#include <cstdint>
#include <string_view>

using namespace __ig;

extern ObjectManager ThreadOM;

FreeValueInfo::FreeValueInfo(uint32_t TypeId, uint32_t Size, char *VPtr)
    : TypeId(TypeId), Size(Size), VPtr(VPtr), MPtr(nullptr) {}
FreeValueInfo::FreeValueInfo(uint32_t TypeId, uint32_t Size, char *VPtr,
                             char *VCmpPtr, size_t CmpSize)
    : TypeId(TypeId), Size(Size), VPtr(VPtr), MPtr(nullptr), VCmpPtr(VCmpPtr),
      MCmpPtr(nullptr), CmpSize(CmpSize) {}

uint32_t FreeValueInfo::getNumValues(FreeValueManager &FVM) const {
  if (!UsesIndirectValues)
    return NumValues;
  return FVM.StringCache.size();
}
char *FreeValueInfo::getValuePtr(FreeValueManager &FVM) const {
  if (!UsesIndirectValues)
    return &Values[Idx * ValueSize];
  return FVM.StringCache[Idx].data();
}
size_t FreeValueInfo::getWrittenSize(FreeValueManager &FVM) const {
  if (!UsesIndirectValues)
    return ValueSize;
  return CmpSize;
}
size_t FreeValueInfo::getValueSize(FreeValueManager &FVM) const {
  if (!UsesIndirectValues)
    return ValueSize;
  return std::min(FVM.StringCache[Idx].size(), CmpSize);
}

bool FreeValueInfo::isFixed() {
  if (IsFixed)
    return true;

  switch (TypeId) {
  case /*memcmp */ 11:
  case /*integer*/ 12:
  case /*pointer*/ 14:
    break;
  default:
    return IsFixed = true;
  }

  if (!isMemcmp()) {
    bool IsInitialized;
    auto *BP = ThreadOM.getBasePtrInfo(VPtr);
    MPtr = ThreadOM.decodeForAccess(VPtr, Size, TypeId, CHECK_INITIALIZED, BP,
                                    IsInitialized);
    if (IsInitialized) {
      return IsFixed = true;
    }
  } else {
    if (CmpSize == 0) {
      // 0-length strings are equal.
      return IsFixed = true;
    }
    bool IsInitialized1 = true, IsInitialized2 = true;
    MPtr = VPtr;
    MCmpPtr = VCmpPtr;
    bool UnknownSize = (CmpSize == (size_t)-1);
    auto *BP1 = ThreadOM.getBasePtrInfo(VPtr);
    if (BP1)
      MPtr = ThreadOM.decodeForAccess(VPtr, UnknownSize ? 1 : CmpSize, TypeId,
                                      CHECK_INITIALIZED, BP1, IsInitialized1);
    auto *BP2 = ThreadOM.getBasePtrInfo(VCmpPtr);
    if (BP2)
      MCmpPtr =
          ThreadOM.decodeForAccess(VCmpPtr, UnknownSize ? 1 : CmpSize, TypeId,
                                   CHECK_INITIALIZED, BP2, IsInitialized2);
    MCmpPtr = VCmpPtr;
    if (IsInitialized1 == IsInitialized2)
      return IsFixed = true;
    if (IsInitialized1) {
      std::swap(MPtr, MCmpPtr);
      std::swap(VPtr, VCmpPtr);
    }
    if (UnknownSize)
      CmpSize = strlen(MCmpPtr) + 1;
  }
  return false;
}

char *FreeValueInfo::write(FreeValueManager &FVM) {
  size_t ValueSize = getValueSize(FVM);
  switch (TypeId) {
  case 2:
    *((float *)MPtr) = 3.14;
    break;
  case 3:
    *((double *)MPtr) = 3.14;
    break;
  case /*TokenTy -> memcmp */ 11: {
    assert(VCmpPtr);
    __builtin_memcpy(MPtr, getValuePtr(FVM), ValueSize);
    if (ValueSize < CmpSize)
      __builtin_memset(MPtr + ValueSize, 0, CmpSize - ValueSize);
    DEBUG("Written '{}' @ {} - {} [{} {}] ({})\n", MPtr, (void *)MPtr,
          (void *)(MPtr + std::max(ValueSize, CmpSize)), ValueSize, CmpSize,
          (void *)VPtr);
    return getValuePtr(FVM);
  }
  case 12:
  case 14:
    __builtin_memcpy(MPtr, getValuePtr(FVM), ValueSize);
    if (ValueSize < Size)
      __builtin_memset(MPtr + ValueSize, 0, Size - ValueSize);
    DEBUG("Written '{}' @ {} - {}\n", *(uint32_t *)MPtr, (void *)MPtr,
          (void *)(MPtr + ValueSize));
    return getValuePtr(FVM);
  default:
    __builtin_memset(MPtr, 0, Size);
  }
  return nullptr;
}

void FreeValueInfo::markInitialized(FreeValueManager &FVM, char *VP,
                                    char *VPC) {
  bool IsInitialized;
  if (auto *BP1 = ThreadOM.getBasePtrInfo(VPtr))
    ThreadOM.decodeForAccess(VPtr, getWrittenSize(FVM), TypeId, BCI_READ, BP1,
                             IsInitialized);
  if (auto *BP1 = ThreadOM.getBasePtrInfo(VCmpPtr))
    ThreadOM.decodeForAccess(VPtr, getWrittenSize(FVM), TypeId, BCI_READ, BP1,
                             IsInitialized);
}

void FreeValueManager::checkBranchConditions(char *VP, char *VPBP, char *VCP,
                                             char *VCPBP) {

  BCISetTy SeenBCIs;
  BCIVecTy FreeBCIs;
  FreeValueVecTy FVVec;

  auto CollectBCIs = [&](char *VPtr) {
    if (auto *BCIVec = lookupBCIVec(VPtr))
      for (auto *BCI : *BCIVec) {
        if (BCI->IsFixed)
          continue;
        if (!SeenBCIs.insert(BCI).second)
          continue;
        bool HasFreeValues = false;
        for (auto &FVI : BCI->FreeValueInfos) {
          if (isFreeValue(*BCI, FVI)) {
            HasFreeValues = true;
            FVVec.push_back(&FVI);
          }
        }
        if (HasFreeValues) {
          FreeBCIs.push_back(BCI);
        } else {
          BCI->IsFixed = true;
          if (!evaluate(*BCI)) {
            INFO("Inconsistent branch condition found, abort\n");
            error(1007);
          }
        }
      }
  };
  if (VP)
    CollectBCIs(VP);
  if (VCP)
    CollectBCIs(VCP);

  if (FreeBCIs.empty())
    return;

  for (auto *FVI : FVVec)
    FVI->write(*this);

  DEBUG("Got {} free BCIs out of {} total, checking\n", FreeBCIs.size(),
        SeenBCIs.size());
  bool AllWork = true;
  for (auto *BCI : FreeBCIs) {
    bool Result = evaluate(*BCI, FreeBCIs.size() == 10);
    if (Result)
      continue;
    AllWork = false;
    break;
  }

  if (!AllWork && !workOn(FVVec, FreeBCIs)) {
    INFO("Could not make all BCIs work, abort\n");
    error(1008);
  }

  for (auto *FVI : FVVec)
    FVI->markInitialized(*this, VP, VCP);
}

bool FreeValueManager::isFreeValue(BranchConditionInfo &BCI,
                                   FreeValueInfo &FVI) {
  if (FVI.isFixed())
    return false;

  if (FVI.isInitialized())
    return true;

  // Initialize the new FVD.
  switch (FVI.TypeId) {
  case /*TokenTy -> memcmp */ 11: {
    assert(FVI.VCmpPtr);
    FVI.UsesIndirectValues = true;
    // TODO: this is not good we should register globals instead.
    auto [It, New] = StringCacheSet.insert(FVI.MCmpPtr);
    if (New) {
      StringCache.push_back(FVI.MCmpPtr);
      auto ItNode = StringCacheSet.extract(It);
      ItNode.value() = std::string_view(StringCache.back());
      StringCacheSet.insert(std::move(ItNode));
    }
    break;
  }
  case 12: {
    FVI.setValues(I32Values, NumI32Values);
    break;
  }
  case 14: {
    FVI.setValues(PtrValues, 1);
    break;
  }
  default:
    ERR("unexpected type id: {}\n", FVI.TypeId);
    __builtin_trap();
  }

  // Write the initial value into the argument buffer.
  return true;
}

bool FreeValueManager::workOn(FreeValueVecTy &FVVec, BCIVecTy &FreeBCIs) {
  for (auto *FVIPtr : FVVec) {
    for (uint32_t I = 0, E = FVIPtr->getNumValues(*this); I < E; ++I) {
      if (modifyAndEvaluate(*FVIPtr, FreeBCIs))
        return true;
      for (auto *OtherFVIPtr : FVVec) {
        if (FVIPtr == OtherFVIPtr)
          continue;
        for (uint32_t J = 0, E = OtherFVIPtr->getNumValues(*this); J < E; ++J)
          if (modifyAndEvaluate(*OtherFVIPtr, FreeBCIs))
            return true;
      }
    }
  }

  return false;
}

bool FreeValueManager::modifyAndEvaluate(FreeValueInfo &FVI,
                                         BCIVecTy &FreeBCIs) {
  // Change the value by chaning the index.
  FVI.Idx = (++FVI.Idx) % FVI.getNumValues(*this);
  FVI.write(*this);

  // Evaluate all BCIs.
  return std::all_of(FreeBCIs.begin(), FreeBCIs.end(),
                     [&](BranchConditionInfo *BCI) { return evaluate(*BCI); });
}

bool FreeValueManager::evaluate(BranchConditionInfo &BCI, bool B) {
  uint32_t Outcome = BCI.Fn(BCI.ArgMemPtr);
  uint32_t DesiredOutcome = ThreadOM.getDesiredOutcome(BCI.No);
  VERBOSE("BCI {}: {} vs {}\n", BCI.No, Outcome, DesiredOutcome);
  return (Outcome == DesiredOutcome);
}

void FreeValueManager::reset() {
  for (auto *BCIPtr : BranchConditionMap) {
    delete[] BCIPtr->ArgMemPtr;
    delete BCIPtr;
  }
  BranchConditionMap.clear();
  BranchConditions.clear();
}

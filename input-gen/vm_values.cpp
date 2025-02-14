
#include "vm_values.h"
#include "vm_obj.h"

#include <cstdint>

using namespace __ig;

extern thread_local ObjectManager ThreadOM;

bool FreeValueDecisionTy::isConsistent(FreeValueInfo &FVI, char *Ptr1,
                                       char *Ptr2) {
  auto Result = __builtin_memcmp(Ptr1, Ptr2, FVI.getManifestSize());
  printf("Comparing %p : %p : %zu, got %i\n", Ptr1, Ptr2, FVI.getManifestSize(),
         Result);
  if (FVI.TypeId != 11)
    return !Result;
  Result = (Result > 0 ? 1 : (Result < 0 ? -1 : 0));
  printf("Comparing '%s' : '%s' :: Idx %i Res %i --> %i\n", Ptr1, Ptr2, Idx,
         Result, Result == ((int32_t)Idx - 1) * (IsSwappedMemcmp ? -1 : 1));
  return Result == ((int32_t)Idx - 1) * (IsSwappedMemcmp ? -1 : 1);
}

bool FreeValueDecisionTy::isConsistent(FreeValueInfo &FVI,
                                       FreeValueDecisionTy &OtherFVD,
                                       FreeValueInfo &OtherFVI) {
  char *MPtr = getSrcMPtr(FVI);
  char *OtherMPtr = OtherFVD.getSrcMPtr(OtherFVI);
  if (!FVI.isMemcmp() && !OtherFVI.isMemcmp()) {
    assert(FVI.VPtr == OtherFVI.VPtr && "TODO offsets");
    return isConsistent(FVI, MPtr, OtherMPtr);
  }

  // TODO offsets
  assert(getDstVPtr(FVI) == OtherFVD.getDstVPtr(OtherFVI) && "TODO offsets");
  if (FVI.VPtr == OtherFVI.VPtr || FVI.VCmpPtr == OtherFVI.VCmpPtr)
    return isConsistent(FVI, OtherMPtr, MPtr);
  return isConsistent(FVI, MPtr, OtherMPtr);
}

bool FreeValueDecisionTy::isFixed(FreeValueInfo &FVI, bool &Consistent) {
  if (IsFixed)
    return true;
  switch (FVI.TypeId) {
  case /*TokenTy -> memcmp */ 11:
  case 12:
    break;
  default:
    return IsFixed = true;
  }

  bool IsMemcmp = FVI.VCmpPtr;
  if (!IsMemcmp) {
    bool IsInitialized;
    char *MPtr =
        ThreadOM.decodeAndCheckInitialized(FVI.VPtr, FVI.Size, IsInitialized);
    printf("LOAD %i : %p\n", IsInitialized, FVI.VPtr);
    if (IsInitialized) {
      Consistent &= isConsistent(FVI, MPtr, getSrcMPtr(FVI));
      return IsFixed = true;
    }
  } else {
    if (FVI.CmpSize == 0) {
      // 0-length strings are equal.
      Idx = 1;
      return IsFixed = true;
    }
    bool IsInitialized1, IsInitialized2;
    char *MPtr1 = ThreadOM.decodeAndCheckInitialized(FVI.VPtr, FVI.CmpSize,
                                                     IsInitialized1);
    char *MPtr2 = ThreadOM.decodeAndCheckInitialized(FVI.VCmpPtr, FVI.CmpSize,
                                                     IsInitialized2);
    printf("MEMCMP %i %i : %p : %p\n", IsInitialized1, IsInitialized2, FVI.VPtr,
           FVI.VCmpPtr);
    if (IsInitialized1 == IsInitialized2) {
      Consistent &= isConsistent(FVI, MPtr1, MPtr2);
      return IsFixed = true;
    }
    IsSwappedMemcmp = IsInitialized1;
  }
  return false;
}

char *FreeValueDecisionTy::write(char *DstMPtr, FreeValueInfo &FVI,
                                 bool IsManifest) {
  if (IsManifest && !willManifest(FVI))
    return nullptr;

  switch (FVI.TypeId) {
  case 2:
    *((float *)DstMPtr) = 3.14;
    break;
  case 3:
    *((double *)DstMPtr) = 3.14;
    break;
  case /*TokenTy -> memcmp */ 11: {
    assert(FVI.VCmpPtr && FVI.Size == sizeof(int));
    if (!IsManifest) {
      __builtin_memcpy(DstMPtr, getValuePtr(), ValueSize);
      return getValuePtr();
    }
    auto *SrcMPtr = getSrcMPtr(FVI);
    __builtin_memcpy(DstMPtr, SrcMPtr, FVI.CmpSize);
    return SrcMPtr;
  }
  case 12:
    __builtin_memcpy(DstMPtr, getValuePtr(), ValueSize);
    if (ValueSize < FVI.Size)
      __builtin_memset(DstMPtr + ValueSize, 0, FVI.Size - ValueSize);
    return getValuePtr();
  case 14:
    *((void **)DstMPtr) = 0;
    break;
  default:
    __builtin_memset(DstMPtr, 0, FVI.Size);
  }
  return nullptr;
}

char *FreeValueDecisionTy::getSrcMPtr(FreeValueInfo &FVI) {
  auto *SrcVPtr = getSrcVPtr(FVI);
  if (auto *SrcBasePtrInfo = ThreadOM.getBasePtrInfo(SrcVPtr))
    return ThreadOM.decodeForAccess(SrcVPtr, FVI.Size, FVI.TypeId, TEST,
                                    SrcBasePtrInfo);
  return SrcVPtr;
}

void FreeValueDecisionTy::manifest(BranchConditionInfo &BCI,
                                   FreeValueInfo &FVI) {
  auto *VPtr = getDstVPtr(FVI);
  printf("Manifest BCI %u -> %u\n", BCI.No, Idx);
  auto *MPtr = ThreadOM.decodeForAccess(VPtr, FVI.Size, FVI.TypeId, TEST_READ,
                                        ThreadOM.getBasePtrInfo(VPtr));
  auto *SrcMPtr = write(MPtr, FVI, /*IsManifest=*/true);

  switch (FVI.TypeId) {
  case 11:
    printf("Wrote %i (%s) to %p\n", *(int *)MPtr, SrcMPtr, FVI.VPtr);
    break;
  case 12:
    printf("Wrote %i (%i) to %p\n", *(int *)MPtr, *((int *)SrcMPtr), FVI.VPtr);
    break;
  default:
    break;
  }
}

void FreeValueManager::checkBranchConditions(char *VP, char *VCP) {
  BCISetTy BCIs;

  auto CollectBCIs = [&](char *VPtr) {
    if (auto *BCIVec = lookupBCIVec(VPtr))
      BCIs.insert(BCIVec->begin(), BCIVec->end());
  };
  CollectBCIs(VP);
  if (VCP)
    CollectBCIs(VCP);

  if (BCIs.empty())
    return;

  BCISetTy BadBCIs;
  DecisionInfluenceMapTy DecisionInfluenceMap;
  FreeDecisionMapTy FreeDecisionMap;
  WriteMapTy WriteMap;

  for (auto *BCI : BCIs) {
    for (auto &FVI : BCI->FreeValueInfos) {
      bool Consistent = true;
      if (auto *FVD = identifyFreeValueDecision(*BCI, FVI, Consistent)) {
        DecisionInfluenceMap[FVD].push_back({BCI, &FVI});
        FreeDecisionMap[BCI].push_back({&FVI, FVD});

        auto *DstVPtr = FVD->getDstVPtr(FVI);
        for (uint32_t I = 0, E = FVI.getManifestSize(); I < E; ++I)
          WriteMap[DstVPtr + I].push_back({FVD, &FVI, BCI});
      }

      if (!Consistent) {
        fprintf(stderr, "Inconsistent fixed value decision found, abort\n");
        error(103);
      }
    }
  }

  printf("Got %zu BCIs, checking\n", BCIs.size());
  for (auto *BCI : BCIs) {
    bool Result = evaluate(*BCI);
    if (Result)
      continue;
    if (FreeDecisionMap[BCI].empty())
      error(101);
    BadBCIs.insert(BCI);
  }

  std::set<FreeValueInfo *> SeenFVIs;
  for (auto [VPtr, It] : WriteMap) {
    if (It.size() < 2)
      continue;
    uint32_t ManifestIdx = ~0;
    for (uint32_t I = 0, E = It.size(); I < E; ++I) {
      auto [FVDPtr, FVIPtr, BCIPtr] = It[I];
      if (!FVDPtr->willManifest(*FVIPtr))
        continue;
      ManifestIdx = I;
      break;
    }
    if (ManifestIdx == ~0U)
      continue;
    SeenFVIs.clear();
    auto [FVDPtr, FVIPtr, BCIPtr] = It[ManifestIdx];
    for (uint32_t I = 0, E = It.size(); I < E; ++I) {
      if (I == ManifestIdx)
        continue;
      auto [OtherFVDPtr, OtherFVIPtr, OtherBCIPtr] = It[I];
      if (!SeenFVIs.insert(OtherFVIPtr).second)
        continue;
      if (!OtherFVDPtr->isConsistent(*OtherFVIPtr, *FVDPtr, *FVIPtr)) {
        BadBCIs.insert(BCIPtr);
        BadBCIs.insert(OtherBCIPtr);
      }
    }
  }

  while (!BadBCIs.empty()) {
    auto *BadBCI = *BadBCIs.begin();
    printf("Got %zu BadBCIs; work on %u\n", BadBCIs.size(), BadBCI->No);
    BadBCIs.erase(BadBCIs.begin());
    if (!workOn(*BadBCI, FreeDecisionMap, DecisionInfluenceMap, WriteMap,
                BadBCIs)) {
      error(102);
    }
  }

  printf("No more bad BCIs, manifest decisions\n");
  for (auto *BCIPtr : BCIs)
    for (auto [FVIPtr, FVDPtr] : FreeDecisionMap[BCIPtr])
      FVDPtr->manifest(*BCIPtr, *FVIPtr);
}

FreeValueDecisionTy *FreeValueManager::identifyFreeValueDecision(
    BranchConditionInfo &BCI, FreeValueInfo &FVI, bool &Consistent) {
  auto &FVD = getOrCreateFreeValueDecision(FVI);
  if (FVD.isInitialized()) {
    // Write the fixed value into the argument buffer.
    FVD.write(BCI, FVI);
    if (FVD.isFixed(FVI, Consistent))
      return nullptr;
    return &FVD;
  }

  // Initialize the new FVD.
  switch (FVI.TypeId) {
  case /*TokenTy -> memcmp */ 11: {
    assert(FVI.VCmpPtr && FVI.Size == sizeof(int));
    FVD.setValues(MemcmpValues, NumMemcmpValues);
    break;
  }
  case 12: {
    FVD.setValues(I32Values, NumI32Values);
    break;
  }
  default:
    fprintf(stderr, "unexpected type id\n");
    __builtin_trap();
  }

  // Write the initial value into the argument buffer.
  FVD.write(BCI, FVI);
  return &FVD;
}

bool FreeValueManager::workOn(BranchConditionInfo &BCI,
                              FreeDecisionMapTy &FreeDecisionMap,
                              DecisionInfluenceMapTy &DecisionInfluenceMap,
                              WriteMapTy &WriteMap, BCISetTy &BadBSIs) {
  const auto &BCIFreeDecisionsVec = FreeDecisionMap[&BCI];
  for (auto [FVIPtr, FVDPtr] : BCIFreeDecisionsVec) {
    for (uint32_t I = 0, E = FVDPtr->NumValues; I < E; ++I) {
      if (modifyAndEvaluate(*FVDPtr, *FVIPtr, DecisionInfluenceMap, WriteMap,
                            BadBSIs))
        return true;
      for (auto [OtherFVIPtr, OtherFVDPtr] : BCIFreeDecisionsVec) {
        if (OtherFVDPtr == FVDPtr)
          continue;
        for (uint32_t J = 0, E = FVDPtr->NumValues; J < E; ++J)
          if (modifyAndEvaluate(*OtherFVDPtr, *OtherFVIPtr,
                                DecisionInfluenceMap, WriteMap, BadBSIs))
            return true;
      }
    }
  }

  return false;
}

bool FreeValueManager::modifyAndEvaluate(
    FreeValueDecisionTy &FVD, FreeValueInfo &FVI,
    DecisionInfluenceMapTy &DecisionInfluenceMap, WriteMapTy &WriteMap,
    BCISetTy &BadBSIs) {
  // Change the value by chaning the index.
  FVD.Idx = (++FVD.Idx) % FVD.NumValues;

  // Evaluate all impacted BCIs.
  for (auto [BCIPtr, FVIPtr] : DecisionInfluenceMap[&FVD]) {
    printf("Modify BCU %u, offset %u uses idx: %u\n", BCIPtr->No,
           FVIPtr->Offset, FVD.Idx);
    FVD.write(*BCIPtr, *FVIPtr);
    if (!evaluate(*BCIPtr))
      return false;
  }

  return checkConsistency(FVD, FVI, WriteMap, BadBSIs);
}

bool FreeValueManager::checkConsistency(FreeValueDecisionTy &FVD,
                                        FreeValueInfo &FVI,
                                        WriteMapTy &WriteMap,
                                        BCISetTy &BadBSIs) {
  std::set<FreeValueInfo *> SeenFVIs;
  char *DstVPtr = FVD.getDstVPtr(FVI);
  bool WillManifest = FVD.willManifest(FVI);
  for (uint32_t Offset = 0; Offset < FVI.getManifestSize(); ++Offset) {
    auto It = WriteMap[DstVPtr + Offset];
    if (It.size() < 2)
      continue;
    for (uint32_t I = 0, E = It.size(); I < E; ++I) {
      auto [OtherFVDPtr, OtherFVIPtr, OtherBCIPtr] = It[I];
      if (OtherFVIPtr == &FVI || !SeenFVIs.insert(OtherFVIPtr).second)
        continue;
      printf("WM %i : OWM %i\n", WillManifest,
             OtherFVDPtr->willManifest(*OtherFVIPtr));
      // TODO Offset is missing
      if (OtherFVDPtr->willManifest(*OtherFVIPtr)) {
        if (!FVD.isConsistent(FVI, *OtherFVDPtr, *OtherFVIPtr))
          return false;
      } else if (WillManifest &&
                 !OtherFVDPtr->isConsistent(*OtherFVIPtr, FVD, FVI))
        BadBSIs.insert(OtherBCIPtr);
    }
  }

  return true;
}

bool FreeValueManager::evaluate(BranchConditionInfo &BCI) {
  char Outcome = BCI.Fn(BCI.ArgMemPtr);
  char DesiredOutcome = ThreadOM.getDesiredOutcome(BCI.No);
  printf("BCI %u: %i vs %i\n", BCI.No, Outcome, DesiredOutcome);
  return (Outcome == DesiredOutcome);
}

void FreeValueManager::reset() {
  for (auto &It : FreeValueDecisions)
    delete It.second;
  FreeValueDecisions.clear();
  std::set<BranchConditionInfo *> FreedBCIs;
  for (auto &[VPtr, BCIs] : BranchConditions)
    for (auto *BCIPtr : BCIs)
      if (FreedBCIs.insert(BCIPtr).second)
        delete BCIPtr;
  BranchConditions.clear();
}

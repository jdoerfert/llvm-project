#ifndef VM_OBJ_H
#define VM_OBJ_H

#include <algorithm>
#include <bit>
#include <cassert>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <functional>
#include <ios>
#include <list>
#include <map>
#include <random>
#include <string_view>
#include <sys/types.h>
#include <tuple>
#include <type_traits>
#include <unordered_set>

#include "logging.h"
#include "vm_choices.h"
#include "vm_enc.h"

namespace __ig {
using BucketScheme10Ty = BucketSchemeTy</*EncodingNo=*/0,
                                        /*OffsetBits=*/12, /*BucketBits=*/3,
                                        /*RealPtrBits=*/32>;
using TableScheme20Ty = TableSchemeTy<1, 30>;

struct ObjectManager {
  ~ObjectManager();

  ObjectManager() : UserBS10(*this), RTObjs(*this), Distribution(-100, 128) {}

  ChoiceTrace *CT = nullptr;
  BucketScheme10Ty UserBS10;
  TableScheme20Ty RTObjs;

  std::string ProgramName;
  std::function<void(uint32_t)> StopFn;

  std::mt19937 Generator;
  std::uniform_int_distribution<int32_t> Distribution;

  void init(ChoiceTrace *CT, std::string_view ProgramName,
            std::function<void(uint32_t)> StopFn) {
    this->CT = CT;
    this->ProgramName = ProgramName;
    this->StopFn = StopFn;
  }
  void setSeed(uint32_t Seed) { Generator.seed(Seed); }

  int32_t getRandomNumber() { return Distribution(Generator); }

  void saveInput(uint32_t InputIdx, uint32_t ExitCode);
  void reset();

  void *getObj(uint32_t Seed);
  char *encode(char *Ptr, uint32_t Size) { return UserBS10.encode(Ptr, Size); }

  std::tuple<char *, uint32_t, uint32_t> decode(char *VPtr) {
    switch (getEncoding(VPtr)) {
    case 0:
      return UserBS10.decode(VPtr);
    case 1:
      return RTObjs.decode(VPtr);
    default:
      return {VPtr, 0, 0};
    }
  }

  __attribute__((always_inline)) char *
  decodeForAccess(char *VPtr, uint32_t AccessSize, uint32_t TypeId,
                  AccessKind AK, char *BasePtrInfo) {
    switch ((uint64_t)BasePtrInfo) {
    case 0:
      return UserBS10.access(VPtr, AccessSize, TypeId, AK == WRITE);
    case 1:
      bool IsInitialized;
      if (AK == READ)
        checkBranchConditions(VPtr);
      return RTObjs.access(VPtr, AccessSize, TypeId, AK, IsInitialized);
    default:
      ERR("unknown encoding {}\n", getEncoding(VPtr));
      UserBS10.error(6);
      std::terminate();
    }
  }

  int32_t getEncoding(char *VPtr) {
    switch (EncodingSchemeTy::getEncoding(VPtr)) {
    case 0:
      return UserBS10.isEncoded(VPtr) ? 0 : ~0;
    case 1:
      return RTObjs.isEncoded(VPtr) ? 1 : ~0;
    default:
      return ~0;
    }
  }

  char *add(char *Addr, int32_t Size, uint32_t Seed) {
    return RTObjs.create(Size, Seed);
  }

  std::pair<int32_t, int32_t> getPtrInfo(char *VPtr, bool AllowToFail) {
    switch (getEncoding(VPtr)) {
    case 0:
      return UserBS10.getPtrInfo(VPtr);
    case 1:
      return RTObjs.getPtrInfo(VPtr);
    default:
      if (AllowToFail)
        return {-2, -2};
      ERR("unknown encoding {}\n", getEncoding(VPtr));
      UserBS10.error(7);
      std::terminate();
    }
  }
  char *getBasePtrInfo(char *VPtr) {
    switch (getEncoding(VPtr)) {
    case 0:
      return UserBS10.getBasePtrInfo(VPtr);
    case 1:
      return RTObjs.getBasePtrInfo(VPtr);
    default:
      ERR("unknown encoding {}\n", getEncoding(VPtr));
      UserBS10.error(8);
      std::terminate();
    }
  }

  bool comparePtrs(bool CmpResult, char *LHSPtr, int32_t LHSInfo,
                   uint32_t LHSOffset, char *RHSPtr, int32_t RHSInfo,
                   uint32_t RHSOffset) {
    if (LHSInfo == RHSInfo) {
      // TODO: Learn from the pointer offset about future runs.
      return CmpResult;
    }

    auto TryToMakeObjNull = [&](char *Obj, TableSchemeBaseTy::TableEntryTy &TE,
                                uint32_t Offset) {
      if (TE.AnyAccess)
        return CmpResult;
      if (TE.IsNull)
        return !CmpResult;
      //      if (CT->addBooleanChoice()) {
      //        TE.IsNull = true;
      //        return !CmpResult;
      //      }
      return CmpResult;
    };
    auto *LHSTE = LHSInfo >= 0 ? &RTObjs.Table[LHSInfo] : nullptr;
    auto *RHSTE = RHSInfo >= 0 ? &RTObjs.Table[RHSInfo] : nullptr;
    if (LHSPtr == 0 && RHSInfo > 0)
      return TryToMakeObjNull(RHSPtr, *RHSTE, RHSOffset);
    if (RHSPtr == 0 && LHSInfo > 0)
      return TryToMakeObjNull(LHSPtr, *LHSTE, LHSOffset);

    if (LHSInfo < 0 || RHSInfo < 0) {
      ERR("comparison of user object and runtime object! C/C++ UB detected! "
          "({}[{}] {}[{}])\n",
          LHSInfo, LHSOffset, RHSInfo, RHSOffset);
      UserBS10.error(43);
      std::terminate();
    }

    // Merge objects or
    return CmpResult;
  }

  uint64_t ptrToInt(char *VPtr, uint64_t Value) {
    auto [PtrInfo, PtrOffset] = getPtrInfo(VPtr, /*AllowToFail=*/true);
    if (PtrInfo >= 0) {
      auto &TE = RTObjs.Table[PtrInfo];
      if (TE.IsNull)
        return 0;
      if (TE.AnyAccess)
        return Value;
      //      if (CT->addBooleanChoice()) {
      //        TE.IsNull = true;
      //        return 0;
      //      }
    }
    return Value;
  }

  char *decodeAndCheckInitialized(char *VPtr, uint32_t Size,
                                  bool &Initialized) {
    switch (getEncoding(VPtr)) {
    case 0:
      Initialized = true;
      return std::get<0>(RTObjs.decode(VPtr));
    case 1:
      Initialized = false;
      return RTObjs.access(VPtr, Size, 0, TEST, Initialized);
    default:
      Initialized = true;
      return VPtr;
    }
  }

  bool getDesiredOutcome(uint32_t ChoiceNo) {
    return CT->addBooleanChoice(ChoiceNo);
  }

  struct BranchConditionInfo {
    struct FreeValueInfo {
      uint16_t Offset;
      uint16_t TypeId;
      uint32_t Size;
      char *VPtr;
    };

    std::vector<FreeValueInfo> FreeValueInfos;
    uint32_t No;
    using FnTy = char (*)(void *);
    FnTy Fn;
    char *ArgMemPtr;
  };

  std::map<char *, std::list<BranchConditionInfo *>> BranchConditions;

  void checkBranchConditions(char *VPtr) {
    auto &BCIs = BranchConditions[VPtr];
    if (BCIs.empty())
      return;

    int32_t I32Values[] = {-100, -64, -32, -8,  -4,  -3,  -2,  -1, 0,
                           1,    2,   3,   4,   8,   12,  16,  22, 24,
                           26,   32,  64,  128, 256, 512, 1024};
    uint32_t NumValues = sizeof(I32Values) / sizeof(I32Values[0]);

    std::set<BranchConditionInfo *> BadBCIs;
    std::map<char *, uint32_t> VIdxMap;
    std::map<char *,
             std::vector<std::pair<BranchConditionInfo *,
                                   BranchConditionInfo::FreeValueInfo *>>>
        VPtrBCIMap;

    auto SetValue = [&](BranchConditionInfo &BCI,
                        BranchConditionInfo::FreeValueInfo &FVI) {
      auto *ArgPtr = BCI.ArgMemPtr + FVI.Offset;
      switch (FVI.TypeId) {
      case 2:
        *((float *)ArgPtr) = 3.14;
        return false;
        break;
      case 3:
        *((double *)ArgPtr) = 3.14;
        return false;
        break;
      case 12: {
        bool IsInitialized;
        auto *MPtr =
            decodeAndCheckInitialized(FVI.VPtr, FVI.Size, IsInitialized);
        if (IsInitialized) {
          __builtin_memcpy(ArgPtr, MPtr, FVI.Size);
          return false;
          break;
        }
        uint32_t VIdx = getRandomNumber() % NumValues;
        auto [It, New] = VIdxMap.insert({FVI.VPtr, VIdx});
        if (New) {
        } else {
          VIdx = It->second;
        }
        __builtin_memcpy(ArgPtr, &I32Values[VIdx], std::min(FVI.Size, 4u));
        if (FVI.Size > 4)
          __builtin_memset(ArgPtr + 4, 0, FVI.Size - 4);
        return true;
        break;
      }
      case 14:
        *((void **)ArgPtr) = 0;
        return false;
        break;
      default:
        __builtin_memset(ArgPtr, 0, FVI.Size);
        return false;
      }
    };

    auto EvaluateBCI = [&](BranchConditionInfo &BCI) {
      char Outcome = BCI.Fn(BCI.ArgMemPtr);
      char DesiredOutcome = getDesiredOutcome(BCI.No);
      return Outcome == DesiredOutcome;
    };

    auto IncAndEvaluate = [&](char *VPtr,
                              std::set<BranchConditionInfo *> &TestedBCIs,
                              std::set<BranchConditionInfo *> &BrokenBCIs) {
      TestedBCIs.clear();
      BrokenBCIs.clear();
      auto &VIdx = VIdxMap[VPtr];
      VIdx = (++VIdx % NumValues);
      auto BCIs = VPtrBCIMap[VPtr];
      for (auto [BCI, FVI] : BCIs) {
        TestedBCIs.insert(BCI);
        SetValue(*BCI, *FVI);
        if (!EvaluateBCI(*BCI))
          BrokenBCIs.insert(BCI);
      }
      return BrokenBCIs.empty();
    };

    auto WorkOnBCI = [&](BranchConditionInfo &BCI) {
      std::set<BranchConditionInfo *> AllBCIs;
      AllBCIs.insert(&BCI);

      std::vector<BranchConditionInfo::FreeValueInfo *> FVIs;
      for (auto &FVI : BCI.FreeValueInfos)
        if (VIdxMap.count(FVI.VPtr))
          FVIs.push_back(&FVI);
      for (uint32_t I = 0; I < FVIs.size(); ++I) {
        for (auto [OtherBCI, _] : VPtrBCIMap[FVIs[I]->VPtr]) {
          if (AllBCIs.insert(OtherBCI).second)
            for (auto &OtherFVI : OtherBCI->FreeValueInfos)
              if (VIdxMap.count(OtherFVI.VPtr))
                FVIs.push_back(&OtherFVI);
        }
      }

      std::set<BranchConditionInfo *> TestedBCIs, BrokenBCIs;
      auto TestAllBCIs = [&]() {
        for (auto *BCI : AllBCIs) {
          if (TestedBCIs.count(BCI))
            continue;
          if (!EvaluateBCI(*BCI))
            return false;
        }
        for (auto *BCI : AllBCIs)
          BadBCIs.erase(BCI);
        return true;
      };

      std::set<char *> VPtrSet;
      for (auto *FVI : FVIs) {
        if (!VPtrSet.insert(FVI->VPtr).second)
          continue;
        for (uint32_t I = 0; I < NumValues; ++I) {
          if (IncAndEvaluate(FVI->VPtr, TestedBCIs, BrokenBCIs) &&
              TestAllBCIs())
            return true;
          for (auto *OtherFVI : FVIs) {
            if (OtherFVI->VPtr == FVI->VPtr)
              continue;
            for (uint32_t J = 0; J < NumValues; ++J)
              if (IncAndEvaluate(OtherFVI->VPtr, TestedBCIs, BrokenBCIs) &&
                  TestAllBCIs())
                return true;
          }
        }
      }
      return false;
    };

    for (auto *BCI : BCIs) {
      uint32_t NumFreeLoads = 0;
      for (auto &FVI : BCI->FreeValueInfos) {
        VPtrBCIMap[FVI.VPtr].push_back({BCI, &FVI});
        NumFreeLoads += SetValue(*BCI, FVI);
      }
      if (EvaluateBCI(*BCI))
        continue;
      if (!NumFreeLoads) {
        UserBS10.error(101);
      }
      BadBCIs.insert(BCI);
    }

    while (!BadBCIs.empty()) {
      auto *BadBCI = *BadBCIs.begin();
      BadBCIs.erase(BadBCIs.begin());
      if (!WorkOnBCI(*BadBCI)) {
        UserBS10.error(102);
      }
    }

    for (auto [VPtr, VIdx] : VIdxMap) {
      auto BCIs = VPtrBCIMap[VPtr];
      // TODO the FVIs for different BCIs could be different (size, typeid etc).
      auto *FVI = BCIs.front().second;
      if (FVI->TypeId == 12) {
        bool IsInitialized;
        auto *MPtr = RTObjs.access(VPtr, FVI->Size, FVI->TypeId, TEST_READ,
                                   IsInitialized);
        auto &VIdx = VIdxMap[FVI->VPtr];
        __builtin_memcpy(MPtr, &I32Values[VIdx], std::min(FVI->Size, 4u));
        if (FVI->Size > 4)
          __builtin_memset(MPtr + 4, 0, FVI->Size - 4);
      }
    }
  }
};

} // namespace __ig
#endif

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
#include "vm_values.h"

namespace __ig {
using BucketScheme10Ty = BucketSchemeTy</*EncodingNo=*/1,
                                        /*OffsetBits=*/12, /*BucketBits=*/3,
                                        /*RealPtrBits=*/32>;
using TableScheme20Ty = TableSchemeTy<2, 30>;

struct ObjectManager {
  ~ObjectManager();

  ObjectManager() : UserBS10(*this), RTObjs(*this), Distribution(-100, 128) {}

  ChoiceTrace *CT = nullptr;
  BucketScheme10Ty UserBS10;
  TableScheme20Ty RTObjs;

  std::string ProgramName;

  std::mt19937 Generator;
  std::uniform_int_distribution<int32_t> Distribution;

  void init(ChoiceTrace *CT, std::string_view ProgramName,
            std::function<void(uint32_t)> StopFn) {
    this->CT = CT;
    this->ProgramName = ProgramName;
    printf("Init %p : %i\n", &ErrorFn, !!StopFn);
    ErrorFn = StopFn;
    printf("Init %p : %i\n", &ErrorFn, !!ErrorFn);
  }
  void setSeed(uint32_t Seed) { Generator.seed(Seed); }

  int32_t getRandomNumber() { return Distribution(Generator); }

  void saveInput(uint32_t InputIdx, uint32_t ExitCode);
  void reset();

  void *getObj(uint32_t Seed);
  char *encode(char *Ptr, uint32_t Size) { return UserBS10.encode(Ptr, Size); }

  std::tuple<char *, uint32_t, uint32_t> decode(char *VPtr) {
    switch (getEncoding(VPtr)) {
    case 1:
      return UserBS10.decode(VPtr);
    case 2:
      return RTObjs.decode(VPtr);
    default:
      return {VPtr, 0, 0};
    }
  }

  __attribute__((always_inline)) char *
  decodeForAccess(char *VPtr, uint32_t AccessSize, uint32_t TypeId,
                  AccessKind AK, char *BasePtrInfo,
                  bool CheckBranchCond = true) {
    switch ((uint64_t)BasePtrInfo) {
    case 1:
      return UserBS10.access(VPtr, AccessSize, TypeId, AK == WRITE);
    case 2:
      bool IsInitialized;
      if (AK == READ && CheckBranchCond)
        checkBranchConditions(VPtr);
      return RTObjs.access(VPtr, AccessSize, TypeId, AK, IsInitialized);
    default:
      ERR("unknown encoding {}\n", getEncoding(VPtr));
      error(6);
      std::terminate();
    }
  }

  int32_t getEncoding(char *VPtr) {
    switch (EncodingSchemeTy::getEncoding(VPtr)) {
    case 1:
      return UserBS10.isEncoded(VPtr) ? 1 : ~0;
    case 2:
      return RTObjs.isEncoded(VPtr) ? 2 : ~0;
    default:
      return ~0;
    }
  }

  char *add(char *Addr, int32_t Size, uint32_t Seed) {
    return RTObjs.create(Size, Seed);
  }

  std::pair<int32_t, int32_t> getPtrInfo(char *VPtr, bool AllowToFail) {
    switch (getEncoding(VPtr)) {
    case 1:
      return UserBS10.getPtrInfo(VPtr);
    case 2:
      return RTObjs.getPtrInfo(VPtr);
    default:
      if (AllowToFail)
        return {-2, -2};
      ERR("unknown encoding {}\n", getEncoding(VPtr));
      error(7);
      std::terminate();
    }
  }
  char *getBasePtrInfo(char *VPtr) {
    switch (getEncoding(VPtr)) {
    case 1:
      return UserBS10.getBasePtrInfo(VPtr);
    case 2:
      return RTObjs.getBasePtrInfo(VPtr);
    default:
      ERR("unknown encoding {}\n", getEncoding(VPtr));
      // TODO: Workaround until global supported.
      return 0;
      error(8);
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
      error(43);
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
    case 1:
      Initialized = true;
      return std::get<0>(RTObjs.decode(VPtr));
    case 2:
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

  FreeValueManager FVM;

  void checkBranchConditions(char *VP, char *VCP = nullptr) {
    return FVM.checkBranchConditions(VP, VCP);
  }
  void addBranchCondition(char *VPtr, BranchConditionInfo *BCI) {
    FVM.BranchConditions[VPtr].push_back(BCI);
  }
};

} // namespace __ig
#endif

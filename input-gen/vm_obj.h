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
using BigObjScheme10Ty = BigObjSchemeTy</*EncodingNo=*/3, /*ObjectBits=*/10>;

struct ObjectManager {
  ~ObjectManager();

  ObjectManager()
      : UserObjLarge(*this), UserObjSmall(*this), RTObjs(*this),
        Distribution(-100, 128) {}

  ChoiceTrace *CT = nullptr;
  BigObjScheme10Ty UserObjLarge;
  BucketScheme10Ty UserObjSmall;
  TableScheme20Ty RTObjs;

  std::string ProgramName;

  std::mt19937 Generator;
  std::uniform_int_distribution<int32_t> Distribution;

  void init(ChoiceTrace *CT, std::string_view ProgramName,
            std::function<void(uint32_t)> StopFn) {
    this->CT = CT;
    this->ProgramName = ProgramName;
    ErrorFn = StopFn;
  }
  void setSeed(uint32_t Seed) { Generator.seed(Seed); }

  int32_t getRandomNumber() { return Distribution(Generator); }

  void saveInput(uint32_t InputIdx, uint32_t ExitCode);
  void reset();

  void *getObj(uint32_t Seed);
  char *encode(char *Ptr, uint32_t Size) {
    if (Size < (1 << 10))
      return UserObjSmall.encode(Ptr, Size);
    return UserObjLarge.encode(Ptr, Size);
  }

  char *decode(char *VPtr) {
    switch (getEncoding(VPtr)) {
    case 1:
      return UserObjSmall.decode(VPtr);
    case 2:
      return RTObjs.decode(VPtr);
    case 3:
      return UserObjLarge.decode(VPtr);
    default:
      return VPtr;
    }
  }

  __attribute__((always_inline)) char *
  decodeForAccess(char *VPtr, uint32_t AccessSize, uint32_t TypeId,
                  AccessKind AK, char *BasePtrInfo, bool &IsInitialized) {
    switch ((uint64_t)BasePtrInfo) {
    case 1:
      IsInitialized = true;
      return UserObjSmall.access(VPtr, AccessSize, TypeId, AK == WRITE);
    case 2:
      IsInitialized = false;
      return RTObjs.access(VPtr, AccessSize, TypeId, AK, IsInitialized);
    case 3:
      IsInitialized = true;
      return UserObjLarge.access(VPtr, AccessSize, TypeId, AK == WRITE);
    default:
      WARN("unknown encoding {} (allowed until global support)\n",
           getEncoding(VPtr));
      // TODO: Workaround until global supported.
      return VPtr;
      ERR("unknown encoding {}\n", getEncoding(VPtr));
      error(1003);
      std::terminate();
    }
  }

  int32_t getEncoding(char *VPtr) {
    switch (EncodingSchemeTy::getEncoding(VPtr)) {
    case 1:
      return UserObjSmall.isEncoded(VPtr) ? 1 : ~0;
    case 2:
      return RTObjs.isEncoded(VPtr) ? 2 : ~0;
    case 3:
      return UserObjLarge.isEncoded(VPtr) ? 3 : ~0;
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
      return UserObjSmall.getPtrInfo(VPtr);
    case 2:
      return RTObjs.getPtrInfo(VPtr);
    case 3:
      return UserObjLarge.getPtrInfo(VPtr);
    default:
      if (AllowToFail)
        return {-2, -2};
      WARN("unknown encoding {} (allowed until global support)\n",
           getEncoding(VPtr));
      // TODO: Workaround until global supported.
      return {-2, -2};
      ERR("unknown encoding {}\n", getEncoding(VPtr));
      error(1004);
      std::terminate();
    }
  }
  char *getBasePtrInfo(char *VPtr) {
    switch (getEncoding(VPtr)) {
    case 1:
      return UserObjSmall.getBasePtrInfo(VPtr);
    case 2:
      return RTObjs.getBasePtrInfo(VPtr);
    case 3:
      return UserObjLarge.getBasePtrInfo(VPtr);
    default:
      WARN("unknown encoding {} (allowed until global support)\n",
           getEncoding(VPtr));
      // TODO: Workaround until global supported.
      return 0;
      error(1005);
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
      error(1006);
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

  bool checkRange(char *VPtr, uint32_t Size) {
    switch (getEncoding(VPtr)) {
    case 1:
      return UserObjSmall.checkSize(VPtr, Size);
    case 2: {
      bool IsInitialized = false;
      RTObjs.access(VPtr, Size, 0, CHECK_INITIALIZED, IsInitialized);
      return IsInitialized;
    }
    case 3:
      return UserObjLarge.checkSize(VPtr, Size);
    default:
      return true;
    }
  }

  char *decodeAndCheckInitialized(char *VPtr, uint32_t Size,
                                  bool &Initialized) {
    switch (getEncoding(VPtr)) {
    case 1:
      Initialized = true;
      return UserObjSmall.decode(VPtr);
    case 2:
      Initialized = false;
      return RTObjs.access(VPtr, Size, 0, CHECK_INITIALIZED, Initialized);
    case 3:
      Initialized = true;
      return UserObjLarge.decode(VPtr);
    default:
      Initialized = true;
      return VPtr;
    }
  }

  bool getDesiredOutcome(uint32_t ChoiceNo) {
    return CT->addBooleanChoice(ChoiceNo);
  }

  FreeValueManager FVM;

  void checkBranchConditions(char *VP, char *VPBP, char *VCP = nullptr,
                             char *VCPBP = nullptr) {
    return FVM.checkBranchConditions(VP, VPBP, VCP, VCPBP);
  }
  void addBranchCondition(char *VPtr, BranchConditionInfo *BCI) {
    FVM.BranchConditionMap[BCI->No] = BCI;
    FVM.BranchConditions[VPtr].push_back(BCI);
  }
  BranchConditionInfo *getBranchCondition(uint32_t No) {
    return FVM.BranchConditionMap[No];
  }
};

} // namespace __ig
#endif

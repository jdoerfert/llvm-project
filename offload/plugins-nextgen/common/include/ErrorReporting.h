//===- ErrorReporting.h - Helper to provide nice error messages ----- c++ -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef OFFLOAD_PLUGINS_NEXTGEN_COMMON_ERROR_REPORTING_H
#define OFFLOAD_PLUGINS_NEXTGEN_COMMON_ERROR_REPORTING_H

#include "PluginInterface.h"
#include "Shared/Environment.h"
#include "Shared/EnvironmentVar.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Frontend/OpenMP/OMP.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <optional>
#include <string>
#include <unistd.h>

namespace llvm {
namespace omp {
namespace target {
namespace plugin {

class ErrorReporter {

  enum ColorTy {
    Yellow = int(HighlightColor::Address),
    Green = int(HighlightColor::String),
    DarkBlue = int(HighlightColor::Tag),
    Cyan = int(HighlightColor::Attribute),
    DarkPurple = int(HighlightColor::Enumerator),
    DarkRed = int(HighlightColor::Macro),
    BoldRed = int(HighlightColor::Error),
    BoldLightPurple = int(HighlightColor::Warning),
    BoldDarkGrey = int(HighlightColor::Note),
    BoldLightBlue = int(HighlightColor::Remark),
  };

  /// The banner printed at the beginning of an error report.
  static constexpr auto ErrorBanner = "OFFLOAD ERROR: ";

  static constexpr uint64_t InvalidLocationId = -1;
  static constexpr uint64_t AmbiguousCallLocationId = -2;
  static constexpr uint64_t InvalidLineOrColumn = -1;

  /// Return the device id as string, or n/a if not available.
  static std::string getDeviceIdStr(GenericDeviceTy *Device) {
    return Device ? std::to_string(Device->getDeviceId()) : "n/a";
  }

  /// Return a nice name for an TargetAllocTy.
  static StringRef getAllocTyName(TargetAllocTy Kind) {
    switch (Kind) {
    case TARGET_ALLOC_DEVICE_NON_BLOCKING:
    case TARGET_ALLOC_DEFAULT:
    case TARGET_ALLOC_DEVICE:
      return "device memory";
    case TARGET_ALLOC_HOST:
      return "pinned host memory";
    case TARGET_ALLOC_SHARED:
      return "managed memory";
      break;
    }
    llvm_unreachable("Unknown target alloc kind");
  }

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgcc-compat"
#pragma clang diagnostic ignored "-Wformat-security"
  /// Print \p Format, instantiated with \p Args to stderr.
  /// TODO: Allow redirection into a file stream.
  template <typename... ArgsTy>
  [[gnu::format(__printf__, 1, 2)]] static void print(const char *Format,
                                                      ArgsTy &&...Args) {
    raw_fd_ostream OS(STDERR_FILENO, false);
    OS << llvm::format(Format, Args...);
  }

  /// Print \p Format, instantiated with \p Args to stderr, but colored.
  /// TODO: Allow redirection into a file stream.
  template <typename... ArgsTy>
  [[gnu::format(__printf__, 2, 3)]] static void
  print(ColorTy Color, const char *Format, ArgsTy &&...Args) {
    raw_fd_ostream OS(STDERR_FILENO, false);
    WithColor(OS, HighlightColor(Color)) << llvm::format(Format, Args...);
  }

  /// Print \p Format, instantiated with \p Args to stderr, but colored and with
  /// a banner.
  /// TODO: Allow redirection into a file stream.
  template <typename... ArgsTy>
  [[gnu::format(__printf__, 1, 2)]] static void reportError(const char *Format,
                                                            ArgsTy &&...Args) {
    print(BoldRed, "%s", ErrorBanner);
    print(BoldRed, Format, Args...);
    print("\n");
  }

  /// Print \p Format, instantiated with \p Args to stderr, but colored with
  /// a banner.
  template <typename... ArgsTy>
  [[gnu::format(__printf__, 1, 2)]] static void
  reportWarning(const char *Format, ArgsTy &&...Args) {
    print(Yellow, "WARNING: ");
    print(Yellow, Format, Args...);
  }
#pragma clang diagnostic pop

  static void reportError(const char *Str) { reportError("%s", Str); }
  static void print(const char *Str) { print("%s", Str); }
  static void print(StringRef Str) { print("%s", Str.str().c_str()); }
  static void print(ColorTy Color, const char *Str) { print(Color, "%s", Str); }
  static void print(ColorTy Color, StringRef Str) {
    print(Color, "%s", Str.str().c_str());
  }

  static void reportLocation(SanitizerEnvironmentTy &SE) {
    print(BoldLightPurple,
          "Triggered by thread <%u,%u,%u> block <%u,%u,%u> PC %p\n",
          SE.ThreadId[0], SE.ThreadId[1], SE.ThreadId[2], SE.BlockId[0],
          SE.BlockId[1], SE.BlockId[2], (void *)SE.PC);
  }

  /// Pretty print a stack trace.
  static void reportStackTrace(StringRef StackTrace) {
    if (StackTrace.empty())
      return;

    SmallVector<StringRef> Lines, Parts;
    StackTrace.split(Lines, "\n", /*MaxSplit=*/-1, /*KeepEmpty=*/false);
    int Start = Lines.empty() || !Lines[0].contains("PrintStackTrace") ? 0 : 1;
    unsigned NumDigits =
        (int)(floor(log10(Lines.size() - Start - /*0*/ 1)) + 1);
    for (int I = Start, E = Lines.size(); I < E; ++I) {
      auto Line = Lines[I];
      Parts.clear();
      Line = Line.drop_while([](char C) { return std::isspace(C); });
      Line.split(Parts, " ", /*MaxSplit=*/2);
      if (Parts.size() != 3 || Parts[0].size() < 2 || Parts[0][0] != '#') {
        print("%s\n", Line.str().c_str());
        continue;
      }
      unsigned FrameIdx = std::stoi(Parts[0].drop_front(1).str());
      if (Start)
        FrameIdx -= 1;
      print(DarkPurple, "    %s", Parts[0].take_front().str().c_str());
      print(Green, "%*u", NumDigits, FrameIdx);
      print(BoldLightBlue, " %s", Parts[1].str().c_str());

      StringRef SignatureAndLocation = Parts[2];
      auto LastSpace = SignatureAndLocation.rfind(' ');
      if (LastSpace != StringRef::npos)
        print(" %s", SignatureAndLocation.substr(0, LastSpace).str().c_str());
      print(Yellow, "%s\n",
            SignatureAndLocation.substr(LastSpace).str().c_str());
    }
    print("\n");
  }

  /// Report information about an allocation associated with \p ATI.
  static void reportAllocationInfo(AllocationTraceInfoTy *ATI) {
    if (!ATI)
      return;

    if (!ATI->DeallocationTrace.empty()) {
      print(BoldLightPurple, "Last deallocation:\n");
      reportStackTrace(ATI->DeallocationTrace);
    }

    if (ATI->HostPtr)
      print(BoldLightPurple,
            "Last allocation of size %lu for host pointer %p -> device pointer "
            "%p:\n",
            ATI->Size, ATI->HostPtr, ATI->DevicePtr);
    else
      print(BoldLightPurple,
            "Last allocation of size %lu -> device pointer %p:\n", ATI->Size,
            ATI->DevicePtr);
    reportStackTrace(ATI->AllocationTrace);
    if (!ATI->LastAllocationInfo)
      return;

    unsigned I = 0;
    print(BoldLightPurple, "Prior allocations with the same base pointer:");
    while (ATI->LastAllocationInfo) {
      print("\n");
      ATI = ATI->LastAllocationInfo;
      print(BoldLightPurple, " #%u Prior deallocation of size %lu:\n", I,
            ATI->Size);
      reportStackTrace(ATI->DeallocationTrace);
      if (ATI->HostPtr)
        print(
            BoldLightPurple,
            " #%u Prior allocation for host pointer %p -> device pointer %p:\n",
            I, ATI->HostPtr, ATI->DevicePtr);
      else
        print(BoldLightPurple, " #%u Prior allocation -> device pointer %p:\n",
              I, ATI->DevicePtr);
      reportStackTrace(ATI->AllocationTrace);
      ++I;
    }
  }

  /// End the execution of the program.
  static void abortExecution() { abort(); }

  static std::pair<const char *, uint32_t> asToString(uint32_t AS) {
    switch (AS) {
    case 0:
      return {"generic", strlen("generic")};
    case 1:
      return {"global", strlen("global")};
    case 3:
      return {"shared", strlen("shared")};
    case 4:
      return {"constant", strlen("constant")};
    case 5:
      return {"stack", strlen("stack")};
    default:
      return {"", 0};
    }
  }

  static void printFakePointer(GenericDeviceTy &Device, DeviceImageTy &Image,
                               SanitizerEnvironmentTy &SE) {
    uint32_t AS = SE.FP.Enc32.RealAS;
    bool Is32Bit = AS == 3 || AS == 5;

    char FakePtrBits[68]{};
    auto ASLeadingZeros = std::min(__builtin_clzg((uintptr_t)SE.FP.VPtr), 3);
    memset(&FakePtrBits[0], '0', ASLeadingZeros);
    auto ToCharResult = std::to_chars(&FakePtrBits[ASLeadingZeros],
                                      &FakePtrBits[64 + ASLeadingZeros],
                                      (uintptr_t)SE.FP.VPtr, 2);
    if (ToCharResult.ec != std::errc()) {
      REPORT("WARNING: %s\n",
             std::make_error_code(ToCharResult.ec).message().c_str());
      return;
    }

    auto [ASStr, ASStrLength] = asToString(AS);
    if (!ASStrLength) {
      REPORT("WARNING: Invalid address space for fake pointer, abort\n");
      return;
    }

    auto PrintFakePtrBits = [&](ColorTy Color, uint32_t First,
                                uint32_t NumBits) {
      char FakePtrBitsTmp[65]{};
      memcpy(&FakePtrBitsTmp[0], &FakePtrBits[First], NumBits);
      print(Color, FakePtrBitsTmp);
      return First + NumBits;
    };

    const char *EmptyString = "";
    const char *LeftMessage = "encoding: ";
    int32_t MessageOffset = strlen(LeftMessage);
    print("\n");
    int32_t Offset = MessageOffset + 4 - (ASStrLength + 7) / 2;
    print(BoldDarkGrey, "%*s%s memory", Offset, EmptyString, ASStr);

    if (Is32Bit) {
      Offset = 32 - (ASStrLength + 7) / 2;
      print(BoldLightPurple, "%*s|alloc. size|", Offset, EmptyString);
      Offset = 3;
      print(BoldLightBlue, "%*s|ptr. offset|\n", Offset, EmptyString);
      print("%s0b", LeftMessage);
      uint32_t Idx = 0;
      // RealAS
      Idx = PrintFakePtrBits(BoldDarkGrey, Idx, 3);
      // RealPtr
      Idx = PrintFakePtrBits(Cyan, Idx, 32);
      // Size
      Idx = PrintFakePtrBits(BoldLightPurple, Idx, FAKE_PTR_BASE_BITS_OFFSET);
      // Magic
      Idx = PrintFakePtrBits(BoldDarkGrey, Idx, 3);
      // Offset
      Idx = PrintFakePtrBits(BoldLightBlue, Idx, FAKE_PTR_BASE_BITS_OFFSET);
      print("\n");
      print(Cyan, "%*s|-    real device pointer     -|", MessageOffset + 5,
            EmptyString);
      print(BoldDarkGrey, "%*smagic\n\n", FAKE_PTR_BASE_BITS_OFFSET - 1,
            EmptyString);
    } else {
      Offset = 16 - (ASStrLength + 7) / 2 - 1;
      print(BoldDarkGrey, "%*smagic\n", Offset, EmptyString);
      print("%s0b", LeftMessage);
      uint32_t Idx = 0;
      // RealAS
      Idx = PrintFakePtrBits(BoldDarkGrey, Idx, 3);
      // Slot no
      Idx = PrintFakePtrBits(Green, Idx, 16);
      // Magic
      Idx = PrintFakePtrBits(BoldDarkGrey, Idx, 3);
      // Offset
      Idx = PrintFakePtrBits(BoldLightBlue, Idx, 42);
      print("\n");
      print(Green, "%*s|- alloc. no. -|", MessageOffset + 5, EmptyString);
      print(BoldLightBlue, "   |-             pointer offset           -|\n\n");
    }
  }

  static void reportOutOfBoundsError(GenericDeviceTy &Device,
                                     DeviceImageTy &Image,
                                     SanitizerEnvironmentTy &SE, bool Event) {
    reportError("execution encountered an out-of-bounds access");

    uint32_t AS = SE.FP.Enc32.RealAS;
    bool Is32Bit = AS == 3 || AS == 5;

    uint64_t Offset = Is32Bit ? SE.FP.Enc32.Offset : SE.FP.Enc64.Offset;
    uint64_t Length = Is32Bit ? SE.FP.Enc32.Size : -1;
    void *DevicePtr = nullptr;
    uint64_t AllocationLocationId = InvalidLocationId;
    if (Is32Bit) {
      DevicePtr = (void *)(uint64_t)SE.FP.Enc32.RealPtr;
      if (!Event)
        Device.getFakeHostPtrGlobalInfo(Image, DevicePtr, AllocationLocationId);
    } else {
      if (!Event)
        Device.getFakeHostPtrInfo(Image, SE.FP.Enc64.SlotId, DevicePtr, Length,
                                  AllocationLocationId);
    }
    void *AccessPtr = utils::advancePtr(DevicePtr, Offset);

    uint32_t AccessKind = SE.AccessSize >> 29;
    uint32_t AccessSize = (SE.AccessSize << 3) >> 3;
    auto AccessKindStr = [](uint32_t Kind) {
      switch (Kind) {
      case 1:
        return "READ";
      case 2:
        return "WRITE";
      case 5:
        return "ATOMIC READ";
      case 6:
        return "ATOMIC WRITE";
      case 7:
        return "ATOMIC READ-WRITE";
      default:
        return "ACCESS";
      }
    };

    print(BoldLightBlue,
          "%s of size %u at %p by thread <%u,%u,%u> block <%u,%u,%u>\n",
          AccessKindStr(AccessKind), AccessSize, (void *)SE.PC, SE.ThreadId[0],
          SE.ThreadId[1], SE.ThreadId[2], SE.BlockId[0], SE.BlockId[1],
          SE.BlockId[2]);
    printLocationIdTrace(Device, Image, SE.LocationId, SE.CallId);

    auto [ASStr, ASStrLength] = asToString(AS);
    print(Cyan,
          "%p is located %lu bytes inside of a %lu-byte %s memory region "
          "[%p,%p)\n\n",
          AccessPtr, Offset, Length, ASStr, DevicePtr,
          utils::advancePtr(DevicePtr, Length));

    if (AllocationLocationId != InvalidLocationId) {
      printLocationIdTrace(Device, Image, AllocationLocationId,
                           /*AmbiguousCallEnc*/ 0, /*IsAllocation*/ true);
    } else if (!Is32Bit)
      reportMemoryAccessError(Device, AccessPtr, StringRef(), /*Abort*/ false);

    printFakePointer(Device, Image, SE);
  }

public:
#define DEALLOCATION_ERROR(Format, ...)                                        \
  reportError(Format, __VA_ARGS__);                                            \
  reportStackTrace(StackTrace);                                                \
  reportAllocationInfo(ATI);                                                   \
  abortExecution();

  static void reportDeallocationOfNonAllocatedPtr(void *DevicePtr,
                                                  TargetAllocTy Kind,
                                                  AllocationTraceInfoTy *ATI,
                                                  std::string &StackTrace) {
    DEALLOCATION_ERROR("deallocation of non-allocated %s: %p",
                       getAllocTyName(Kind).data(), DevicePtr);
  }

  static void reportDeallocationOfDeallocatedPtr(void *DevicePtr,
                                                 TargetAllocTy Kind,
                                                 AllocationTraceInfoTy *ATI,
                                                 std::string &StackTrace) {
    DEALLOCATION_ERROR("double-free of %s: %p", getAllocTyName(Kind).data(),
                       DevicePtr);
  }

  static void reportDeallocationOfWrongPtrKind(void *DevicePtr,
                                               TargetAllocTy Kind,
                                               AllocationTraceInfoTy *ATI,
                                               std::string &StackTrace) {
    DEALLOCATION_ERROR("deallocation requires %s but allocation was %s: %p",
                       getAllocTyName(Kind).data(),
                       getAllocTyName(ATI->Kind).data(), DevicePtr);
#undef DEALLOCATION_ERROR
  }

  static void printLocationIdTrace(GenericDeviceTy &Device,
                                   DeviceImageTy &Image, uint64_t LocationId,
                                   uint64_t AmbiguousCallEnc = 0,
                                   bool IsAllocation = false) {
    if (LocationId == InvalidLocationId) {
      reportError("    no backtrace available\n");
      return;
    }

    GenericGlobalHandlerTy &GHandler = Device.Plugin.getGlobalHandler();
    auto GetImagePtr = [&](GlobalTy &GV, bool Quiet = false) {
      if (auto Err = GHandler.getGlobalMetadataFromImage(Device, Image, GV)) {
        if (Quiet)
          consumeError(std::move(Err));
        else
          REPORT("WARNING: Failed to read backtrace "
                 "(%s)\n",
                 toString(std::move(Err)).data());
        return false;
      }
      return true;
    };

    GlobalTy LocationsGV("__offload_san_locations", -1);
    GlobalTy LocationNamesGV("__offload_san_location_names", -1);
    GlobalTy AmbiguousCallsBitWidthGV("__offload_san_num_ambiguous_calls", -1);
    GlobalTy AmbiguousCallsLocationsGV("__offload_san_ambiguous_calls_mapping",
                                       -1);
    if (GetImagePtr(LocationsGV))
      GetImagePtr(LocationNamesGV);
    GetImagePtr(AmbiguousCallsBitWidthGV, /*Quiet=*/true);
    GetImagePtr(AmbiguousCallsLocationsGV, /*Quiet=*/true);

    if (!LocationsGV.getPtr() || !LocationNamesGV.getPtr()) {
      reportError("    no backtrace available\n");
      return;
    }

    char *LocationNames = LocationNamesGV.getPtrAs<char>();
    LocationEncodingTy *Locations = LocationsGV.getPtrAs<LocationEncodingTy>();
    uint64_t *AmbiguousCallsBitWidth =
        AmbiguousCallsBitWidthGV.getPtrAs<uint64_t>();
    uint64_t *AmbiguousCallsLocations =
        AmbiguousCallsLocationsGV.getPtrAs<uint64_t>();

    if (AmbiguousCallEnc && AmbiguousCallsBitWidth) {
      // Get rid of partial encodings at the end of the AmbiguousCallEnc
      AmbiguousCallEnc <<= (64 % *AmbiguousCallsBitWidth);
      AmbiguousCallEnc >>= (64 % *AmbiguousCallsBitWidth);
    }

    if (IsAllocation) {
      LocationEncodingTy &LE = Locations[LocationId];
      print(Green, "allocation of ");
      print(BoldLightPurple, "'%s'", &LocationNames[LE.FunctionNameIdx]);
      if (strlen(&LocationNames[LE.FileNameIdx])) {
        print(Green, " in ");
        print(Yellow, "%s:%lu", &LocationNames[LE.FileNameIdx], LE.LineNo);
      }
      print("\n");
      assert(LE.ParentIdx == InvalidLocationId);
      return;
    }

    int32_t FrameIdx = 0;
    unsigned NumDigits = 2;
    do {
      LocationEncodingTy &LE = Locations[LocationId];
      print(DarkPurple, "    #");
      print(Green, "%*u", NumDigits, FrameIdx);
      print(" %s", &LocationNames[LE.FunctionNameIdx]);
      print(Yellow, " %s:%lu:%lu\n", &LocationNames[LE.FileNameIdx], LE.LineNo,
            LE.ColumnNo);
      LocationId = LE.ParentIdx;
      FrameIdx++;
      if (LocationId == AmbiguousCallLocationId && AmbiguousCallEnc != 0 &&
          AmbiguousCallsBitWidth && AmbiguousCallsLocations) {
        uint64_t LastAmbiguousCallEnc =
            AmbiguousCallEnc & ((1 << *AmbiguousCallsBitWidth) - 1);
        LocationId = AmbiguousCallsLocations[LastAmbiguousCallEnc - 1];
        AmbiguousCallEnc >>= (*AmbiguousCallsBitWidth);
      }
    } while (LocationId != InvalidLocationId &&
             LocationId != AmbiguousCallLocationId);
    print("\n");
  }

  static void reportMemoryAccessError(GenericDeviceTy &Device, void *DevicePtr,
                                      StringRef ErrorStr, bool Abort) {
    if (!ErrorStr.empty())
      reportError(ErrorStr.data());

    if (!Device.OMPX_TrackAllocationTraces) {
      print(Yellow, "Use '%s=true' to track device allocations\n\n",
            Device.OMPX_TrackAllocationTraces.getName().data());
      if (Abort)
        abortExecution();
      return;
    }
    uintptr_t Distance = false;
    auto *ATI =
        Device.getClosestAllocationTraceInfoForAddr(DevicePtr, Distance);
    if (!ATI) {
      print(Cyan,
            "No host-issued allocations; device pointer %p might be "
            "a global, stack, or shared location\n",
            DevicePtr);
      if (Abort)
        abortExecution();
      return;
    }
    if (!Distance) {
      print(Cyan, "Device pointer %p points into%s host-issued allocation:\n",
            DevicePtr, ATI->DeallocationTrace.empty() ? "" : " prior");
      reportAllocationInfo(ATI);
      if (Abort)
        abortExecution();
      return;
    }

    bool IsClose = Distance < (1L << 29L /*512MB=*/);
    print(Cyan,
          "Device pointer %p does not point into any (current or prior) "
          "host-issued allocation%s.\n",
          DevicePtr,
          IsClose ? "" : " (might be a global, stack, or shared location)");
    //    if (IsClose) {
    print(Cyan,
          "Closest host-issued allocation (distance %" PRIuPTR
          " byte%s; might be by page):\n",
          Distance, Distance > 1 ? "s" : "");
    reportAllocationInfo(ATI);
    //    }
    if (Abort)
      abortExecution();
  }

  /// Report that a kernel encountered a trap instruction.
  static void
  reportTrapInKernel(GenericDeviceTy &Device, KernelTraceInfoRecordTy &KTIR,
                     std::function<bool(void *Queue)> AsyncInfoWrapperMatcher,
                     bool Event = false) {
    assert(AsyncInfoWrapperMatcher && "A matcher is required");

    DeviceImageTy *Image = nullptr;
    SanitizerEnvironmentTy *SE = nullptr;
    for (auto &It : Device.SanitizerEnvironmentMap) {
      if (It.second->ErrorCode == SanitizerEnvironmentTy::NONE)
        continue;
      if (SE)
        reportWarning(
            "Multiple errors encountered, information might be inaccurate.");
      std::tie(Image, SE) = It;
      assert(Image);
    }

    uint32_t Idx = 0;
    for (uint32_t I = 0, E = KTIR.size(); I < E; ++I) {
      auto KTI = KTIR.getKernelTraceInfo(I);
      if (KTI.Kernel == nullptr)
        break;
      // Skip kernels issued in other queues.
      if (KTI.Queue && !(AsyncInfoWrapperMatcher(KTI.Queue)))
        continue;
      Idx = I;
      break;
    }

    auto KTI = KTIR.getKernelTraceInfo(Idx);
    if (KTI.Queue && AsyncInfoWrapperMatcher(KTI.Queue)) {
      auto PrettyKernelName =
          llvm::omp::prettifyFunctionName(KTI.Kernel->getName());
      reportError("Kernel '%s'", PrettyKernelName.c_str());
    }
    assert((!SE || SE->ErrorCode != SanitizerEnvironmentTy::NONE) &&
           "Unexpected sanitizer environment");
    if (!SE) {
      reportError("execution stopped, reason is unknown");
      print(Yellow, "Compile with '-mllvm -amdgpu-enable-offload-sanitizer' "
                    "improved diagnosis\n");
    } else {
      switch (SE->ErrorCode) {
      case SanitizerEnvironmentTy::TRAP:
        reportError("execution interrupted by hardware trap instruction");
        reportLocation(*SE);
        printLocationIdTrace(Device, *Image, SE->LocationId, SE->CallId);
        break;
      case SanitizerEnvironmentTy::UNREACHABLE:
        reportError("execution reached an \"unreachable\" state (likely caused "
                  "by undefined behavior)");
        reportLocation(*SE);
        printLocationIdTrace(Device, *Image, SE->LocationId, SE->CallId);
        break;
      case SanitizerEnvironmentTy::BAD_PTR:
        reportError("execution encountered a garbage pointer");
        reportLocation(*SE);
        printLocationIdTrace(Device, *Image, SE->LocationId, SE->CallId);
        break;
      case SanitizerEnvironmentTy::ALLOCATION_TOO_LARGE:
        reportError("execution encountered an allocation that is too large");
        reportLocation(*SE);
        printLocationIdTrace(Device, *Image, SE->LocationId, SE->CallId);
        break;
      case SanitizerEnvironmentTy::AS_MISMATCH:
        reportError(
            "execution encountered a pointer to the wrong address space");
        reportLocation(*SE);
        printLocationIdTrace(Device, *Image, SE->LocationId, SE->CallId);
        break;
      case SanitizerEnvironmentTy::OUT_OF_BOUNDS:
        reportOutOfBoundsError(Device, *Image, *SE, Event);
        break;
      default:
        reportError(
            "execution stopped, reason is unknown due to invalid error code");
      reportLocation(*SE);
      printLocationIdTrace(Device, *Image, SE->LocationId, SE->CallId);
      }
    }

    if (KTI.Queue && AsyncInfoWrapperMatcher(KTI.Queue)) {
      if (!KTI.LaunchTrace.empty()) {
        print(BoldLightPurple, "Kernel launch trace:\n");
        reportStackTrace(KTI.LaunchTrace);
      } else {
        print(Yellow, "Use '%s=1' to show the stack trace of the kernel\n",
              Device.OMPX_TrackNumKernelLaunches.getName().data());
      }
    }
    if (!Event)
      abort();
  }

  static void checkAndReportError(GenericDeviceTy &Device,
                                  __tgt_async_info *AsyncInfo,
                                  KernelTraceInfoRecordTy *KTIR = nullptr,
                                  bool Event = false) {
    SanitizerEnvironmentTy *SE = nullptr;
    for (auto &It : Device.SanitizerEnvironmentMap) {
      if (It.second->ErrorCode == SanitizerEnvironmentTy::NONE)
        continue;
      if (SE)
        reportWarning(
            "Multiple errors encountered, information might be inaccurate.");
      SE = It.second;
    }
    if (!SE)
      return;

    if (KTIR) {
      reportTrapInKernel(
          Device, *KTIR, [=](void *) { return true; }, Event);
    } else {
      auto KernelTraceInfoRecord =
          Device.KernelLaunchTraces.getExclusiveAccessor();
      reportTrapInKernel(
          Device, *KernelTraceInfoRecord, [=](void *) { return true; }, Event);
    }
  }

  /// Report the kernel traces taken from \p KTIR, up to
  /// OFFLOAD_TRACK_NUM_KERNEL_LAUNCH_TRACES many.
  static void reportKernelTraces(GenericDeviceTy &Device,
                                 KernelTraceInfoRecordTy &KTIR) {
    uint32_t NumKTIs = 0;
    for (uint32_t I = 0, E = KTIR.size(); I < E; ++I) {
      auto KTI = KTIR.getKernelTraceInfo(I);
      if (KTI.Kernel == nullptr)
        break;
      ++NumKTIs;
    }
    if (NumKTIs == 0) {
      print(BoldRed, "No kernel launches known\n");
      return;
    }

    uint32_t TracesToShow =
        std::min(Device.OMPX_TrackNumKernelLaunches.get(), NumKTIs);
    if (TracesToShow == 0) {
      if (NumKTIs == 1)
        print(BoldLightPurple, "Display only launched kernel:\n");
      else
        print(BoldLightPurple, "Display last %u kernels launched:\n", NumKTIs);
    } else {
      if (NumKTIs == 1)
        print(BoldLightPurple, "Display kernel launch trace:\n");
      else
        print(BoldLightPurple,
              "Display %u of the %u last kernel launch traces:\n", TracesToShow,
              NumKTIs);
    }

    for (uint32_t Idx = 0, I = 0; I < NumKTIs; ++Idx) {
      auto KTI = KTIR.getKernelTraceInfo(Idx);
      auto PrettyKernelName =
          llvm::omp::prettifyFunctionName(KTI.Kernel->getName());
      if (NumKTIs == 1)
        print(BoldLightPurple, "Kernel '%s'\n", PrettyKernelName.c_str());
      else
        print(BoldLightPurple, "Kernel %d: '%s'\n", I,
              PrettyKernelName.c_str());
      reportStackTrace(KTI.LaunchTrace);
      ++I;
    }

    if (NumKTIs != 1) {
      print(Yellow,
            "Use '%s=<num>' to adjust the number of shown stack traces (%u "
            "now, up to %zu)\n",
            Device.OMPX_TrackNumKernelLaunches.getName().data(),
            Device.OMPX_TrackNumKernelLaunches.get(), KTIR.size());
    }
    // TODO: Let users know how to serialize kernels
  }
};

} // namespace plugin
} // namespace target
} // namespace omp
} // namespace llvm

#endif // OFFLOAD_PLUGINS_NEXTGEN_COMMON_ERROR_REPORTING_H

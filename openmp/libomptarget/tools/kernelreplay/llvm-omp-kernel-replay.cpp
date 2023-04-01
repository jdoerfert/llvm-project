//===- llvm-omp-kernel-replay.cpp - Replay OpenMP offload kernel ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a command line utility to replay the execution of recorded OpenMP
// offload kernels.
//
//===----------------------------------------------------------------------===//

#include "omptargetplugin.h"
#include "Utilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TimeProfiler.h"

#include <cstdlib>

using namespace llvm;
using namespace llvm::omp::target;

cl::OptionCategory ReplayOptions("llvm-omp-kernel-replay Options");

// InputFilename - The filename to read the json description of the kernel.
static cl::opt<std::string> InputFilename(cl::Positional,
                                          cl::desc("<input kernel json file>"),
                                          cl::Required);

static cl::opt<bool> VerifyOpt(
    "verify",
    cl::desc(
        "Verify device memory post execution against the original output."),
    cl::init(false), cl::cat(ReplayOptions));

static cl::opt<bool> SaveOutputOpt(
    "save-output",
    cl::desc("Save the device memory output of the replayed kernel execution."),
    cl::init(false), cl::cat(ReplayOptions));

static cl::opt<unsigned> NumTeamsOpt("num-teams",
                                     cl::desc("Set the number of teams."),
                                     cl::init(0), cl::cat(ReplayOptions));

static cl::opt<unsigned> NumThreadsOpt("num-threads",
                                       cl::desc("Set the number of threads."),
                                       cl::init(0), cl::cat(ReplayOptions));

static cl::opt<int32_t> DeviceIdOpt("device-id", cl::desc("Set the device id."),
                                    cl::init(-1), cl::cat(ReplayOptions));

// Compares the contents of record/replay files with the defined suffix. Returns
// true when those are identical.
bool verifyMem(std::string &KernelHashName, const std::string Suffix) {
  std::string RecordFileName(KernelHashName + ".original" + Suffix);
  ErrorOr<std::unique_ptr<MemoryBuffer>> OriginalOutputMB =
      MemoryBuffer::getFile( RecordFileName,
                            /* isText */ false,
                            /* RequiresNullTerminator */ false);
  if (!OriginalOutputMB)
    report_fatal_error("Error reading:" + StringRef(RecordFileName) + ".\n"
        "make sure LIBOMPTARGET_RR_SAVE_OUTPUT is set when recording");

  std::string ReplayFileName(KernelHashName + ".replay" + Suffix);
  ErrorOr<std::unique_ptr<MemoryBuffer>> ReplayOutputMB =
      MemoryBuffer::getFile(ReplayFileName,
                            /* isText */ false,
                            /* RequiresNullTerminator */ false);
  if (!ReplayOutputMB)
    report_fatal_error("Error reading:" + StringRef(ReplayFileName) + ".\n"
        "make sure LIBOMPTARGET_RR_SAVE_OUTPUT is set when recording");

  StringRef OriginalOutput = OriginalOutputMB.get()->getBuffer();
  StringRef ReplayOutput = ReplayOutputMB.get()->getBuffer();

  return (OriginalOutput == ReplayOutput);
}

void registerGlobalEntries(const void *BufferPtr, const void *BufferEndPtr,
    SmallVector<__tgt_offload_entry>& KernelEntries) {
  while (BufferPtr != BufferEndPtr) {
    StringRef Name((const char *)BufferPtr);
    BufferPtr = advanceVoidPtr(BufferPtr, Name.size() + 1);

    uint32_t RecordedSize = *((const uint32_t *)(BufferPtr));
    BufferPtr = advanceVoidPtr(BufferPtr, sizeof(uint32_t));

    // Set the host pointer of the global such that
    // we read the value from the recording.
    // TODO: Delete code which reads global values, it is no
    // longer needed
    void *GPtr = const_cast<void *> (BufferPtr);
    KernelEntries.push_back({GPtr, const_cast<char *>(Name.data()),
        RecordedSize, 0, 0});
    BufferPtr = advanceVoidPtr(BufferPtr, RecordedSize);
  }
}

int main(int argc, char **argv) {
  TimeTraceScope TimeScope("Replay");

  cl::HideUnrelatedOptions(ReplayOptions);
  cl::ParseCommandLineOptions(argc, argv, "llvm-omp-kernel-replay\n");

  ErrorOr<std::unique_ptr<MemoryBuffer>> KernelInfoMB =
      MemoryBuffer::getFile(InputFilename, /* isText */ true,
                            /* RequiresNullTerminator */ true);
  if (!KernelInfoMB)
    report_fatal_error("Error reading the kernel info json file");
  Expected<json::Value> JsonKernelInfo =
      json::parse(KernelInfoMB.get()->getBuffer());
  if (auto Err = JsonKernelInfo.takeError())
    report_fatal_error("Cannot parse the kernel info json file");

  auto NumTeamsJson =
      JsonKernelInfo->getAsObject()->getInteger("NumTeamsClause");
  unsigned NumTeams = (NumTeamsOpt > 0 ? NumTeamsOpt : NumTeamsJson.value());
  auto NumThreadsJson =
      JsonKernelInfo->getAsObject()->getInteger("ThreadLimitClause");
  unsigned NumThreads =
      (NumThreadsOpt > 0 ? NumThreadsOpt : NumThreadsJson.value());
  // TODO: Print a warning if number of teams/threads is explicitly set in the
  // kernel info but overriden through command line options.
  auto LoopTripCount =
      JsonKernelInfo->getAsObject()->getInteger("LoopTripCount");
  auto KernelFunc = JsonKernelInfo->getAsObject()->getString("Name");

  SmallVector<void *> TgtArgs;
  SmallVector<ptrdiff_t> TgtArgOffsets;
  auto NumArgs = JsonKernelInfo->getAsObject()->getInteger("NumArgs");
  auto *TgtArgsArray = JsonKernelInfo->getAsObject()->getArray("ArgPtrs");
  for (auto It : *TgtArgsArray)
    TgtArgs.push_back(reinterpret_cast<void *>(It.getAsInteger().value()));
  auto *TgtArgOffsetsArray =
      JsonKernelInfo->getAsObject()->getArray("ArgOffsets");
  for (auto It : *TgtArgOffsetsArray)
    TgtArgOffsets.push_back(
        reinterpret_cast<ptrdiff_t>(It.getAsInteger().value()));

  auto DevMemoryStartValue = JsonKernelInfo->getAsObject()->getInteger("DevMemoryStart");
  if ( DevMemoryStartValue )
    setenv("LIBOMPTARGET_DEVMEM_START",
        std::to_string(reinterpret_cast<intptr_t>(*DevMemoryStartValue)).c_str(), 1);

  auto DeviceMemorySizeJson =
      JsonKernelInfo->getAsObject()->getInteger("DeviceMemorySize");
  // Set device memory size to the ceiling of GB granularity.

  uint64_t DeviceMemorySize =
      std::ceil(DeviceMemorySizeJson.value() / (1024.0 * 1024.0 * 1024.0));

  auto DeviceIdJson = JsonKernelInfo->getAsObject()->getInteger("DeviceId");
  // TODO: Print warning if the user overrides the device id in the json file.
  int32_t DeviceId = (DeviceIdOpt > -1 ? DeviceIdOpt : DeviceIdJson.value());

  setenv("LIBOMPTARGET_REPLAY", "1", 1);
  if (VerifyOpt || SaveOutputOpt)
    setenv("LIBOMPTARGET_RR_SAVE_OUTPUT", "1", 1);

  __tgt_init_all_rtls();

  std::string KernelHashName =
      InputFilename.substr(0, InputFilename.find_last_of('.'));
  ErrorOr<std::unique_ptr<MemoryBuffer>> GlobalsMB =
      MemoryBuffer::getFile(KernelHashName + ".globals", /* isText */ false,
                            /* RequiresNullTerminator */ false);

  if (!GlobalsMB)
    report_fatal_error("Error reading the globals data.");

  SmallVector<__tgt_offload_entry> KernelEntries;
  registerGlobalEntries(GlobalsMB.get()->getBufferStart(),
                        GlobalsMB->get()->getBufferEnd(), KernelEntries);

  __tgt_offload_entry KernelEntry = {nullptr, nullptr, 0, 0, 0};
  std::string KernelEntryName = KernelFunc.value().str();
  KernelEntry.name = const_cast<char *>(KernelEntryName.c_str());
  // Anything non-zero works to uniquely identify the kernel.
  KernelEntry.addr = (void *)0x1;
  KernelEntries.push_back(KernelEntry);

  ErrorOr<std::unique_ptr<MemoryBuffer>> ImageMB =
      MemoryBuffer::getFile(KernelHashName + ".image", /* isText */ false,
                            /* RequiresNullTerminator */ false);
  if (!ImageMB)
    report_fatal_error("Error reading the kernel image.");

  __tgt_device_image DeviceImage;
  DeviceImage.ImageStart = (void *)ImageMB.get()->getBufferStart();
  DeviceImage.ImageEnd = (void *)ImageMB.get()->getBufferEnd();
  DeviceImage.EntriesBegin = KernelEntries.begin();
  DeviceImage.EntriesEnd = KernelEntries.end();

  __tgt_bin_desc Desc;
  Desc.NumDeviceImages = 1;
  Desc.HostEntriesBegin = KernelEntries.begin();
  Desc.HostEntriesEnd = KernelEntries.end();
  Desc.DeviceImages = &DeviceImage;

  // TODO: do we need requires?
  //__tgt_register_requires(/* Flags */1);
  __tgt_register_lib(&Desc);

  // Read in the entire file (hence the "as stream"), since we might lock the
  // memory.
  ErrorOr<std::unique_ptr<MemoryBuffer>> DeviceMemoryMB =
      MemoryBuffer::getFileAsStream(KernelHashName + ".memory");
  if (!DeviceMemoryMB)
    report_fatal_error("Error reading the kernel input device memory.");


  __tgt_target_kernel_replay(
      /* Loc */ nullptr, DeviceId, KernelEntry.addr,
      (void *)DeviceMemoryMB.get()->getBuffer().data(),
      DeviceMemoryMB.get()->getBufferSize(), TgtArgs.data(),
      TgtArgOffsets.data(), NumArgs.value(), NumTeams, NumThreads,
      LoopTripCount.value(), GlobalsMB.get()->getBufferStart(),
      GlobalsMB.get()->getBufferSize());

  __tgt_unregister_lib(&Desc);

  if (VerifyOpt) {
    if ( verifyMem(KernelHashName, ".output" ) )
      outs() << "[llvm-omp-kernel-replay] Replay device memory verified!\n";
    else
      outs() << "[llvm-omp-kernel-replay] Replay device memory failed to "
                "verify!\n";

    if ( verifyMem(KernelHashName, ".globals" ) )
      outs() << "[llvm-omp-kernel-replay] Replay globals memory verified!\n";
    else
      outs() << "[llvm-omp-kernel-replay] Replay globals memory failed to "
                "verify!\n";
  }

  return 0;
}

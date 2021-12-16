//===- GlobalHandler.cpp - Target independent global & env. var handling --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Target independent global handler and environment manager.
//
//===----------------------------------------------------------------------===//

#include "GlobalHandler.h"

#include "DeviceInterface.h"

using namespace llvm::omp::plugin;

int32_t GlobalHandlerTy::moveGlobalBetweenDeviceAndHost(
    GenericDeviceTy &Device, const GlobalTy &HostGlobal, bool Device2Host) {
  GlobalTy DeviceGlobal(HostGlobal.getName(), HostGlobal.getSize());
  int32_t Err = getGlobalMetadataFromDevice(Device, DeviceGlobal);
  if (Err) {
    INFO(OMP_INFOTYPE_DATA_TRANSFER, Device.DeviceId,
         "Failed to read global symbol metadata for '%s' from the device",
         HostGlobal.getName().c_str());
    return Err;
  }

  if (Device2Host)
    Err = Device.memcpyDtoH(HostGlobal.getPtr(), DeviceGlobal.getPtr(),
                            HostGlobal.getSize());
  else
    Err = Device.memcpyHtoD(DeviceGlobal.getPtr(), HostGlobal.getPtr(),
                            HostGlobal.getSize());

  if (Err) {
    INFO(OMP_INFOTYPE_DATA_TRANSFER, Device.DeviceId,
         "Failed to %s %u bytes associated with global symbol '%s' %s "
         "the device",
         Device2Host ? "read" : "write", HostGlobal.getSize(),
         HostGlobal.getName().c_str(), Device2Host ? "from" : "to");
    return Err;
  }

  INFO(OMP_INFOTYPE_DATA_TRANSFER, Device.DeviceId,
       "Successfully %s %u bytes associated with global symbol '%s' %s "
       "the device (%p -> %p)",
       Device2Host ? "read" : "write", HostGlobal.getSize(),
       HostGlobal.getName().c_str(), Device2Host ? "from" : "to",
       DeviceGlobal.getPtr(), HostGlobal.getPtr());
  return OFFLOAD_SUCCESS;
}

int32_t GlobalHandlerTy::getGlobalMetadataFromImage(GenericDeviceTy &Device,
                                                    GlobalTy &ImageGlobal) {
  // TODO: We should wrap ELF handling into a caching object.
  Expected<ELF64LEObjectFile> ELF =
      ELF64LEObjectFile::create(Device.getImageBuffer());
  if (!ELF) {
    INFO(OMP_INFOTYPE_DATA_TRANSFER, Device.DeviceId,
         "Unable to open ELF image.");
    return OFFLOAD_FAIL;
  }

  // Then extract the base address of elf image.
  Expected<uint64_t> StartAddr = ELF.get().getStartAddress();
  if (!StartAddr) {
    INFO(OMP_INFOTYPE_DATA_TRANSFER, Device.DeviceId,
         "Unable to determine ELF start address.");
    return OFFLOAD_FAIL;
  }
  char *ELFStartAddr = reinterpret_cast<char *>(StartAddr.get());

  for (auto &It : ELF.get().symbols()) {
    // Fist check the name, continue if we don't match.
    Expected<StringRef> Name = It.getName();
    if (!Name || !Name.get().equals(ImageGlobal.getName()))
      continue;

    // If we match we will either succeed or fail with retriving the content,
    // either way, the loop is done. First step is to verify the size.
    ImageGlobal.setSize(It.getSize());

    // Then extract the relative offset from the elf image base.
    Expected<uint64_t> Offset = It.getValue();
    if (!Offset) {
      INFO(OMP_INFOTYPE_DATA_TRANSFER, Device.DeviceId,
           "Global symbol '%s' was found in the elf image but address could "
           "not be determined.",
           ImageGlobal.getName().c_str());
      return OFFLOAD_FAIL;
    }
    ImageGlobal.setPtr(ELFStartAddr + Offset.get());

    return OFFLOAD_SUCCESS;
  }

  INFO(OMP_INFOTYPE_DATA_TRANSFER, Device.DeviceId,
       "Global symbol '%s' was not found in the elf image.",
       ImageGlobal.getName().c_str());
  return OFFLOAD_FAIL;
}

int32_t GlobalHandlerTy::readGlobalFromImage(GenericDeviceTy &Device,
                                             const GlobalTy &HostGlobal) {
  GlobalTy ImageGlobal(HostGlobal.getName(), -1);
  int32_t Err = getGlobalMetadataFromImage(Device, ImageGlobal);
  if (Err)
    return Err;

  if (ImageGlobal.getSize() != HostGlobal.getSize()) {
    INFO(OMP_INFOTYPE_DATA_TRANSFER, Device.DeviceId,
         "Global symbol '%s' has %u bytes in the elf image but %u bytes "
         "on the host, abort transfer.",
         HostGlobal.getName().c_str(), ImageGlobal.getSize(),
         HostGlobal.getSize());
    return OFFLOAD_FAIL;
  }

  INFO(OMP_INFOTYPE_DATA_TRANSFER, Device.DeviceId,
       "Global symbol '%s' was found in the elf image and %u bytes will "
       "copied from %p to %p.",
       HostGlobal.getName().c_str(), HostGlobal.getSize(), ImageGlobal.getPtr(),
       HostGlobal.getPtr());
  memcpy(HostGlobal.getPtr(), ImageGlobal.getPtr(), HostGlobal.getSize());
  return OFFLOAD_SUCCESS;
}

//===- GlobalHandler.h - Target independent global & enviroment handling --===//
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

#ifndef LLVM_OPENMP_LIBOMPTARGET_PLUGINS_COMMON_GLOBALHANDLER_GLOBALHANDLER_H
#define LLVM_OPENMP_LIBOMPTARGET_PLUGINS_COMMON_GLOBALHANDLER_GLOBALHANDLER_H

#include <string>

#include "Debug.h"
#include "omptarget.h"
#include "llvm/Object/ELFObjectFile.h"

namespace llvm {
namespace omp {
namespace plugin {

struct GenericDeviceTy;

using namespace llvm::object;

/// Common abstraction for globals that live on the host and device.
/// It simply encapsulates the symbol name, symbol size, and symbol address
/// (which might be host or device depending on the context).
/// TODO: We should probably keep both the host and device pointer in this
/// structure to avoid multiple lookups, e.g. when we write device globals
/// in USM mode after we looked up their device address.
class GlobalTy {
  std::string Name;
  uint32_t Size;
  void *Ptr;

public:
  GlobalTy(const std::string &Name, uint32_t Size, void *Ptr = nullptr)
      : Name(Name), Size(Size), Ptr(Ptr) {}

  const std::string &getName() const { return Name; }
  uint32_t getSize() const { return Size; }
  void *getPtr() const { return Ptr; }

  void setSize(int32_t S) { Size = S; }
  void setPtr(void *P) { Ptr = P; }
};

/// Subclass of GlobalTy that holds the memory for a global of \p Ty.
template <typename Ty> class StaticGlobalTy : public GlobalTy {
  Ty Data;

public:
  template <typename... Args>
  StaticGlobalTy(const std::string &Name, Args &&...args)
      : GlobalTy(Name, sizeof(Ty), &Data),
        Data(Ty{std::forward<Args>(args)...}) {}
  template <typename... Args>
  StaticGlobalTy(const char *Name, Args &&...args)
      : GlobalTy(Name, sizeof(Ty), &Data),
        Data(Ty{std::forward<Args>(args)...}) {}
  template <typename... Args>
  StaticGlobalTy(const char *Name, const char *Suffix, Args &&...args)
      : GlobalTy(std::string(Name) + Suffix, sizeof(Ty), &Data),
        Data(Ty{std::forward<Args>(args)...}) {}

  Ty &getValue() { return Data; }
  const Ty &getValue() const { return Data; }
  void setValue(const Ty &V) { Data = V; }
};

/// Subclass of GlobalTy that holds the memory which may exceed the global type
/// \p Ty.
template <typename Ty> class DynamicGlobalTy : public GlobalTy {
public:
  DynamicGlobalTy(const std::string &Name, uint32_t Size)
      : GlobalTy(Name, Size, malloc(Size)) {}
  DynamicGlobalTy(const char *Name, const char *Suffix, uint32_t Size)
      : GlobalTy(std::string(Name) + Suffix, Size, malloc(Size)) {}
  ~DynamicGlobalTy() { free(getPtr()); }

  Ty &getValue() { return *static_cast<Ty *>(getPtr()); }
  const Ty &getValue() const { return *static_cast<Ty *>(getPtr()); }
  void setValue(const Ty &V) { *getPtr() = V; }
};

/// Helper class to do the heavy lifting when it comes to moving globals between
/// host and device. Through the GenericDeviceTy we access memcpy DtoH and HtoD,
/// which means the only things specialized by the subclass is the retrival of
/// global metadata (size, addr) from the device.
/// \see getGlobalMetadataFromDevice
class GlobalHandlerTy {
  /// Actually move memory between host and device. See readGlobalFromDevice and
  /// writeGlobalToDevice for the interface description.
  int32_t moveGlobalBetweenDeviceAndHost(GenericDeviceTy &Device,
                                         const GlobalTy &HostGlobal,
                                         bool Device2Host);

  /// Get the address and size of a global in the image. Return success
  /// or failure. Address and size are return in \p ImageGlobal, the global name
  /// is passed in \p ImageGlobal.
  int32_t getGlobalMetadataFromImage(GenericDeviceTy &Device,
                                     GlobalTy &ImageGlobal);

public:
  /// Copy the memory associated with a global from the host to its counterpart
  /// on the device. The name, size, and destination are defined by
  /// \p HostGlobal. Return success or failure.
  int32_t writeGlobalToDevice(GenericDeviceTy &Device,
                              const GlobalTy &HostGlobal) {
    return moveGlobalBetweenDeviceAndHost(Device, HostGlobal,
                                          /* Device2Host */ false);
  }

  int32_t writeGlobalToImage(GenericDeviceTy &Device,
                             const GlobalTy &HostGlobal);

  /// Read the memory associated with a global from the image and store it on
  /// the host. The name, size, and destination are defined by \p HostGlobal.
  /// Return success or failure.
  int32_t readGlobalFromImage(GenericDeviceTy &Device,
                              const GlobalTy &HostGlobal);

  /// Get the address and size of a global from the device. Return success
  /// or failure. Address is return in \p DeviceGlobal, the global name and
  /// expected size are passed in \p DeviceGlobal.
  int32_t getGlobalMetadataFromDevice(GenericDeviceTy &Device,
                                      GlobalTy &DeviceGlobal);

  /// Copy the memory associated with a global from the device to its
  /// counterpart on the host. The name, size, and destination are defined by
  /// \p HostGlobal. Return success or failure.
  int32_t readGlobalFromDevice(GenericDeviceTy &Device,
                               const GlobalTy &HostGlobal) {
    return moveGlobalBetweenDeviceAndHost(Device, HostGlobal,
                                          /* Device2Host */ true);
  }
};

} // namespace plugin
} // namespace omp
} // namespace llvm

#endif // LLVM_OPENMP_LIBOMPTARGET_PLUGINS_COMMON_GLOBALHANDLER_GLOBALHANDLER_H

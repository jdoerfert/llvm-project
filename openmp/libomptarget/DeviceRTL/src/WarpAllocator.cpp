//===------- WarpAllocator.cpp - Warp memory allocator ------- C++ -*-========//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#pragma omp begin declare target device_type(nohost)

#include "Debug.h"
#include "Mapping.h"
#include "Memory.h"
#include "State.h"
#include "Synchronization.h"
#include "Types.h"
#include "Utils.h"

using namespace ompx;

char *CONSTANT(omptarget_device_heap_buffer)
    __attribute__((used, retain, weak, visibility("protected")));

size_t CONSTANT(omptarget_device_heap_size)
    __attribute__((used, retain, weak, visibility("protected")));

namespace {
constexpr const size_t Alignment = 16;
constexpr const size_t FirstThreadRatio = 40;
constexpr const size_t SplitThreadhold = Alignment * 16;

template <typename T> T abs(T V) { return V > 0 ? V : -V; }

template <int32_t WARP_SIZE> struct WarpAllocator;

class WarpAllocatorEntry {
  template <int32_t WARP_SIZE> friend struct WarpAllocator;

  /// If Size is less than 0, the entry is allocated (in use).
  int64_t Size = 0;
  /// PrevSize is also supposed to be greater than or equal to 0. When it is 0,
  /// it is the first entry of the buffer.
  int64_t PrevSize = 0;

public:
  bool isFirst() const { return !PrevSize; }

  size_t getSize() const { return abs(Size); }
  void setSize(size_t V) { Size = V; }

  void setPrevSize(WarpAllocatorEntry *Prev) {
    PrevSize = Prev ? Prev->getSize() : 0;
  }

  size_t getUserSize() const { return getSize() - sizeof(WarpAllocatorEntry); }

  // Note: isUsed can not be !isUnused or other way around because when Size is
  // 0, it is uninitialized.
  bool isUsed() const { return Size < 0; }
  bool isUnused() const { return Size > 0; }

  void setUsed() {
    assert(isUnused() && "the entry is in use");
    Size *= -1;
  }
  void setUnused() {
    assert(isUsed() && "the entry is not in use");
    Size *= -1;
  }

  char *getUserPtr() { return reinterpret_cast<char *>(this + 1); }
  char *getEndPtr() { return reinterpret_cast<char *>(getNext()); }

  WarpAllocatorEntry *getPrev() { return utils::advance(this, -PrevSize); }
  WarpAllocatorEntry *getNext() { return utils::advance(this, getSize()); }

  static WarpAllocatorEntry *fromUserPtr(void *Ptr) { return fromPtr(Ptr) - 1; }

  static WarpAllocatorEntry *fromPtr(void *Ptr) {
    return reinterpret_cast<WarpAllocatorEntry *>(Ptr);
  }
};

static_assert(sizeof(WarpAllocatorEntry) == 16, "entry size mismatch");

template <int32_t WARP_SIZE> struct WarpAllocator {
  void init() {
    if (mapping::getWarpId() || mapping::getBlockId())
      return;

    size_t HeapSize = omptarget_device_heap_size;
    size_t FirstThreadHeapSize = HeapSize * FirstThreadRatio / 100;
    size_t OtherThreadHeapSize =
        (HeapSize - FirstThreadHeapSize) / (WARP_SIZE - 1);
    OtherThreadHeapSize = utils::align_down(OtherThreadHeapSize, Alignment);

    for (int I = 0; I < WARP_SIZE; ++I) {
      Entries[I] = nullptr;
      size_t PrivateOffset = OtherThreadHeapSize * I + FirstThreadHeapSize;
      Limits[I] = omptarget_device_heap_buffer + PrivateOffset;
    }
  }

  void *allocate(size_t Size) {
    int32_t TIdInWarp = mapping::getThreadIdInWarp();
    Size = utils::align_up(Size + sizeof(WarpAllocatorEntry), Alignment);

    // Error our early if the requested size is larger than the entire block.
    if (Size > getBlockSize(TIdInWarp))
      return nullptr;

    WarpAllocatorEntry *E = nullptr;
    {
      mutex::LockGuard LG(Locks[TIdInWarp]);

      auto *LastEntry = Entries[TIdInWarp];
      auto *NewWatermark =
          (LastEntry ? LastEntry->getEndPtr() : getBlockBegin(TIdInWarp)) +
          Size;
      if (NewWatermark >= Limits[TIdInWarp]) {
        E = findMemorySlow(Size, TIdInWarp);
      } else {
        E = LastEntry ? LastEntry->getNext()
                      : WarpAllocatorEntry::fromPtr(getBlockBegin(TIdInWarp));
        E->setSize(Size);
        E->setPrevSize(LastEntry);
        Entries[TIdInWarp] = E;
      }
    }

    if (!E)
      return nullptr;

    assert(E->isUnused() && "entry is not set to use properly");

    E->setUsed();

    return E->getUserPtr();
  }

  void deallocate(void *Ptr) {
    WarpAllocatorEntry *E = WarpAllocatorEntry::fromUserPtr(Ptr);
    E->setUnused();

    auto TIdInWarp = mapping::getThreadIdInWarp();

    mutex::LockGuard LG(Locks[TIdInWarp]);
    // Is last entry?
    if (E == Entries[TIdInWarp]) {
      do {
        E = E->getPrev();
      } while (!E->isFirst() && !E->isUsed());
      Entries[TIdInWarp] = E;
    }
  }

  memory::MemoryAllocationInfo getMemoryAllocationInfo(void *P) {
    if (!utils::isInRange(P, omptarget_device_heap_buffer,
                          omptarget_device_heap_size))
      return {};

    auto TIdInWarp = mapping::getThreadIdInWarp();
    for (int I = TIdInWarp; I < TIdInWarp + mapping::getWarpSize(); ++I) {
      if (P < getBlockBegin(I % mapping::getWarpSize()) || P >= getBlockEnd(I))
        continue;

      mutex::LockGuard LG(Locks[I]);
      WarpAllocatorEntry *E = Entries[I];
      if (!E)
        return {};
      if (E->getEndPtr() <= P)
        return {};
      do {
        if (E->getUserPtr() <= P && P < E->getEndPtr()) {
          if (!E->isUsed())
            return {};
          return {E->getUserPtr(), E->getUserSize()};
        }
        E = E->getPrev();
      } while (!E->isFirst());
    }
    return {};
  }

private:
  char *getBlockBegin(int32_t TIdInWarp) const {
    return TIdInWarp ? Limits[TIdInWarp - 1] : omptarget_device_heap_buffer;
  }
  char *getBlockEnd(int32_t TIdInWarp) const { return Limits[TIdInWarp]; }

  size_t getBlockSize(int32_t TIdInWarp) const {
    return getBlockEnd(TIdInWarp) - getBlockBegin(TIdInWarp);
  }

  WarpAllocatorEntry *findMemorySlow(size_t Size, int32_t TIdInWarp) {
    char *Ptr = getBlockBegin(TIdInWarp);
    char *Limit = getBlockEnd(TIdInWarp);

    WarpAllocatorEntry *E = WarpAllocatorEntry::fromPtr(Ptr);
    do {
      if (!E->isUsed() && E->getSize() >= Size)
        break;
      E = E->getNext();
      if (reinterpret_cast<char *>(E) + Size > Limit)
        return nullptr;
    } while (1);

    size_t OldSize = E->getSize();
    if (OldSize - Size >= SplitThreadhold) {
      E->setSize(Size);
      auto *LeftOverE = E->getNext();
      LeftOverE->setPrevSize(E);
      LeftOverE->setSize(OldSize - Size);
    }

    return E;
  }

  WarpAllocatorEntry *Entries[WARP_SIZE];
  char *Limits[WARP_SIZE];
  mutex::TicketLock Locks[WARP_SIZE];
};

WarpAllocator<32> Allocator;

} // namespace

namespace ompx {
namespace memory {
MemoryAllocationInfo getMemoryAllocationInfo(void *P) {
  return Allocator.getMemoryAllocationInfo(P);
}
} // namespace memory
} // namespace ompx

extern "C" {

void *malloc(size_t Size) {
  if (!Size)
    return nullptr;
  void *P = Allocator.allocate(Size);
  assert(P && "allocator out of memory");
  assert(reinterpret_cast<intptr_t>(P) % Alignment == 0 &&
         "misaligned address");
  return P;
}

void free(void *P) {
  if (!P)
    return;
  Allocator.deallocate(P);
}

void *realloc(void *ptr, size_t new_size) {
  void *NewPtr = malloc(new_size);
  if (!NewPtr)
    return nullptr;
  WarpAllocatorEntry *E = WarpAllocatorEntry::fromUserPtr(ptr);
  __builtin_memcpy(NewPtr, ptr, utils::min(E->getUserSize(), new_size));
  free(ptr);
  return NewPtr;
}

void __kmpc_target_init_allocator() {
  Allocator.init();
}
}

#pragma omp end declare target

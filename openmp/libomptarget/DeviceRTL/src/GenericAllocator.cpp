//===------- GenericAllocator.cpp - Generic GPU memory allocator -- C++ -*-===//
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
#include "Memory.h"
#include "Synchronization.h"
#include "Utils.h"

using namespace ompx;

char *CONSTANT(omptarget_device_heap_buffer)
    __attribute__((used, retain, weak, visibility("protected")));

size_t CONSTANT(omptarget_device_heap_size)
    __attribute__((used, retain, weak, visibility("protected")));

namespace {
size_t HeapCurPos = 0;
mutex::TicketLock HeapLock;
mutex::TicketLock AllocationListLock;
mutex::TicketLock FreeListLock;
constexpr const size_t Alignment = 16;

using intptr_t = int64_t;

struct SimpleLinkListNode {
  template <typename T> friend struct SimpleLinkList;

protected:
  SimpleLinkListNode *Prev = nullptr;
  SimpleLinkListNode *Next = nullptr;

  bool operator>(const SimpleLinkListNode &RHS) const {
    const auto This = reinterpret_cast<int64_t>(this);
    const auto RHSThis = reinterpret_cast<int64_t>(&RHS);
    return This > RHSThis;
  }

  SimpleLinkListNode() = default;
  SimpleLinkListNode(const SimpleLinkListNode &) = delete;
  SimpleLinkListNode(SimpleLinkListNode &&) = delete;
};

struct AllocationMetadata final : SimpleLinkListNode {
public:
  size_t getUserSize() const;
  void *getUserAddr() const {
    return const_cast<void *>(reinterpret_cast<const void *>(this + 1));
  }

  size_t getSize() const { return Size; }
  void setSize(size_t V) { Size = V; }

  bool isInRange(void *Ptr) {
    return utils::isInRange(Ptr, this + 1, getUserSize());
  }

  static AllocationMetadata *getFromUserAddr(void *P) {
    return getFromAddr(P) - 1;
  }

  static AllocationMetadata *getFromAddr(void *P) {
    return reinterpret_cast<AllocationMetadata *>(P);
  }

private:
  size_t Size = 0;
  int64_t Reserved = 0;
};

constexpr const size_t AllocationMetadataSize = sizeof(AllocationMetadata);
static_assert(AllocationMetadataSize == 32,
              "expect the metadata size to be 32");

size_t AllocationMetadata::getUserSize() const {
  return Size - AllocationMetadataSize;
}

template <typename T> struct SimpleLinkList {
  struct iterator {
    friend SimpleLinkList;

    T *operator->() { return reinterpret_cast<T *>(Node); }
    T &operator*() { return reinterpret_cast<T &>(*Node); }

    iterator operator++() {
      Node = Node->Next;
      return *this;
    }

    bool operator==(const iterator &RHS) const { return Node == RHS.Node; }
    bool operator!=(const iterator &RHS) const { return !(*this == RHS); }

    iterator next() const {
      iterator Itr;
      Itr.Node = Node->Next;
      return Itr;
    }

  private:
    SimpleLinkListNode *Node = nullptr;
  };

  iterator begin() {
    iterator Itr;
    Itr.Node = Head.Next;
    return Itr;
  }

  iterator end() { return {}; }

  void insert(SimpleLinkListNode *Node) { insertImpl(&Head, Node); }

  void remove(iterator Itr) {
    SimpleLinkListNode *Node = Itr.Node;
    remove(Node);
  }

  void remove(SimpleLinkListNode *Node) { removeImpl(Node); }

  bool empty() const { return Head.Next == nullptr; }

private:
  static void insertImpl(SimpleLinkListNode *Current,
                         SimpleLinkListNode *Node) {
    SimpleLinkListNode *OldNext = Current->Next;
    Node->Prev = Current;
    Node->Next = OldNext;
    if (OldNext)
      OldNext->Prev = Node;
    Current->Next = Node;
  }

  static void removeImpl(SimpleLinkListNode *Node) {
    SimpleLinkListNode *Prev = Node->Prev;
    SimpleLinkListNode *Next = Node->Next;
    Prev->Next = Next;
    if (Next)
      Next->Prev = Prev;
    Node->Prev = nullptr;
    Node->Next = nullptr;
  }

  /// Check if the node is in the list.
  bool checkExist(SimpleLinkListNode *Node) const {
    SimpleLinkListNode *P = Head.Next;
    while (P) {
      if (P == Node)
        return true;
      P = P->Next;
    }
    return false;
  }

  static bool checkSanity(SimpleLinkListNode *Current,
                          SimpleLinkListNode *Next) {
    return Current->Next == Next && (!Next || Next->Prev == Current);
  }

  static bool checkSanity(SimpleLinkListNode *Prev, SimpleLinkListNode *Current,
                          SimpleLinkListNode *Next) {
    return Prev->Next == Current && Current->Next == Next &&
           Current->Prev == Prev && (!Next || Next->Prev == Current);
  }

  static bool checkDangling(SimpleLinkListNode *Node) {
    return Node->Next == nullptr && Node->Prev == nullptr;
  }

  SimpleLinkListNode Head;
};

SimpleLinkList<AllocationMetadata> AllocationList;
SimpleLinkList<AllocationMetadata> FreeList;

constexpr const int SplitRatio = 5;

#ifndef LIBOMPTARGET_MEMORY_PROFILING
struct MemoryProfiler {
  MemoryProfiler(const char *S) {}
  ~MemoryProfiler() {}
};
#else
struct MemoryProfiler : utils::SimpleProfiler {
  MemoryProfiler(const char *S) : utils::SimpleProfiler(S) {}
  ~MemoryProfiler() { utils::SimpleProfiler::~SimpleProfiler(); }
};
#endif
} // namespace

namespace ompx {
namespace memory {
MemoryAllocationInfo getMemoryAllocationInfo(void *P) {
  mutex::LockGuard ALG(AllocationListLock);
  for (auto Itr = AllocationList.begin(); Itr != AllocationList.end(); ++Itr) {
    AllocationMetadata *MD = &(*Itr);
    if (MD->isInRange(P))
      return {MD->getUserAddr(), MD->getUserSize()};
  }
  return {};
}

void init() {}
} // namespace memory
} // namespace ompx

extern "C" {

void *memset(void *dest, int ch, size_t count);

void *malloc(size_t Size) {
  MemoryProfiler Profiler(__FUNCTION__);

  Size = utils::align_up(Size + AllocationMetadataSize, Alignment);

  AllocationMetadata *MD = nullptr;

  {
    mutex::LockGuard FLG(FreeListLock);

    auto Itr = FreeList.begin();
    for (; Itr != FreeList.end(); ++Itr) {
      if (Itr->getSize() >= Size)
        break;
    }

    bool Found = Itr != FreeList.end() && (Itr->getSize() / Size < SplitRatio);
    if (Found) {
      MD = &(*Itr);
      FreeList.remove(Itr);
    }
  }

  if (MD) {
    mutex::LockGuard ALG(AllocationListLock);
    AllocationList.insert(MD);
    return MD->getUserAddr();
  }

  {
    mutex::LockGuard LG(HeapLock);

    if (Size + HeapCurPos < omptarget_device_heap_size) {
      void *R = omptarget_device_heap_buffer + HeapCurPos;
      (void)atomic::add(&HeapCurPos, Size, atomic::acq_rel);
      MD = AllocationMetadata::getFromAddr(R);
    }
  }

  if (MD) {
    // We need to reset the head in case of any dirty data.
    memset(MD, 0, AllocationMetadataSize);
    MD->setSize(Size);
    mutex::LockGuard ALG(AllocationListLock);
    AllocationList.insert(MD);
    return MD->getUserAddr();
  }

  printf("out of heap memory! size=%lu, cur=%lu.\n", Size, HeapCurPos);

  printf("%s:%d\n", __FILE__, __LINE__);
  __builtin_trap();
}

void free(void *P) {
  MemoryProfiler Profiler(__FUNCTION__);

  if (!P)
    return;

  auto *MD = AllocationMetadata::getFromUserAddr(P);

  {
    mutex::LockGuard ALG(AllocationListLock);
    AllocationList.remove(MD);
  }

  {
    mutex::LockGuard FLG(FreeListLock);
    FreeList.insert(MD);
  }
}

void *realloc(void *ptr, size_t new_size) {
  MemoryProfiler Profiler(__FUNCTION__);

  void *NewPtr = malloc(new_size);
  if (!NewPtr)
    return nullptr;
  auto *OldMD = AllocationMetadata::getFromUserAddr(ptr);
  assert(ptr == OldMD->getUserAddr());
  __builtin_memcpy(NewPtr, ptr, utils::min(OldMD->getUserSize(), new_size));
  free(ptr);
  return NewPtr;
}
}

#pragma omp end declare target

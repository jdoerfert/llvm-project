
#include "vm_storage.h"
#include "common.h"
#include "global_manager.h"
#include "vm_obj.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <type_traits>

using namespace __ig;

// INPUTGEN
using FileMagicTy = uint64_t;
static constexpr FileMagicTy FileMagic = 0x4e45475455504e49;

static void FATAL() {
  ERR("Malformed input!\n");
  exit(1);
}

template <typename T> static const char *ccast(const T *Ptr) {
  return reinterpret_cast<const char *>(Ptr);
}

template <typename T> static char *ccast(T *Ptr) {
  return reinterpret_cast<char *>(Ptr);
}

template <typename T> static T readV(std::ifstream &Input) {
  if (Input.eof())
    FATAL();
  T El;
  Input.read(ccast(&El), sizeof(El));
  if (Input.gcount() != sizeof(El))
    FATAL();
  return El;
}

static void READMEM(std::ifstream &Input, char *Mem, size_t Size) {
  if (Input.eof())
    FATAL();
  Input.read(Mem, Size);
  if ((size_t)Input.gcount() != Size)
    FATAL();
}

template <typename T> static T writeV(std::ofstream &Output, T El) {
  Output.write(ccast(&El), sizeof(El));
  return El;
}

#ifndef NDEBUG
#define DEFINE_READV(S)                                                        \
  do {                                                                         \
  } while (0)
#define DEFINE_WRITEV(S)                                                       \
  do {                                                                         \
  } while (0)
#define READV(V)                                                               \
  do {                                                                         \
    V = readV<decltype(V)>(IFS);                                               \
    DEBUG("read " #V " {}\n", V);                                              \
  } while (0)
#define WRITEV(V)                                                              \
  do {                                                                         \
    writeV<decltype(V)>(OFS, V);                                               \
    DEBUG("wrote " #V " {}\n", V);                                             \
  } while (0)
#else
#define DEFINE_READV(S) auto READV = [&S]<typename T>(T &V) { V = readV<T>(S); }
#define DEFINE_WRITEV(S) auto WRITEV = [&S]<typename T>(T V) { writeV(S, V); };
#endif

namespace __ig::storage {

struct GlobalComp {
  bool operator()(const GlobalManager::GlobalTy &G, const std::string &Name) {
    return G.Name < Name;
  }
  bool operator()(const std::string &Name, const GlobalManager::GlobalTy &G) {
    return Name < G.Name;
  }
};

Global::Global(std::ifstream &IFS, GlobalManager &GM) {
  DEFINE_READV(IFS);
  uint32_t NameSize;
  READV(NameSize);
  Name.resize(NameSize);
  READMEM(IFS, Name.data(), NameSize);
  auto Found = std::lower_bound(GM.Globals.begin(), GM.Globals.end(), Name,
                                GlobalComp{});
  if (Found == GM.Globals.end() || Found->Name != Name) {
    ERR("Could not find global with name {}\n", Name);
    exit(1003);
  }
  R.init(Range(IFS, Found->Address));
}

void Global::write(std::ofstream &OFS) {
  DEFINE_WRITEV(OFS);
  uint32_t NameSize = Name.size();
  WRITEV(NameSize);
  OFS.write(Name.data(), NameSize);
  R->write(OFS);
}

Range::Range(std::ifstream &IFS, char *Memory) {
  DEFINE_READV(IFS);
  READV(ObjIdx);
  READV(NegativeSize);
  READV(AnyRecorded);
  ptrdiff_t Length;
  READV(Length);
  // TODO I am unsure if this offset it right
  Begin = Memory + NegativeSize;
  End = Begin + Length;
  if (Length) {
    if (AnyRecorded) {
      assert(Memory);
      READMEM(IFS, Begin, Length);
    }
  } else {
    Begin = End = nullptr;
  }
}

Range::Range(std::ifstream &IFS) {
  DEFINE_READV(IFS);
  READV(ObjIdx);
  READV(NegativeSize);
  READV(AnyRecorded);
  ptrdiff_t Length;
  READV(Length);
  if (Length) {
    Begin = (char *)malloc(Length);
    DEBUG("malloc -- {}\n", (void *)Begin);
    End = Begin + Length;
    if (AnyRecorded) {
      READMEM(IFS, Begin, Length);
      INPUTGEN_DEBUG({
        if (getenv("PRINT_RUNTIME_OBJECTS"))
          dumpMemoryHex(Begin, Length);
      });
    }
  } else {
    Begin = End = nullptr;
  }
}

void Range::write(std::ofstream &OFS) {
  DEFINE_WRITEV(OFS);
  WRITEV(ObjIdx);
  WRITEV(NegativeSize);
  WRITEV(AnyRecorded);
  ptrdiff_t Length = End - Begin;
  WRITEV(Length);
  if (Length)
    if (AnyRecorded)
      OFS.write(Begin, Length);
}

Ptr::Ptr(std::ifstream &IFS) {
  DEFINE_READV(IFS);
  READV(ObjIdx);
  READV(Offset);
  READV(TgtObjIdx);
  READV(TgtOffset);
}

void Ptr::write(std::ofstream &OFS) {
  DEFINE_WRITEV(OFS);
  WRITEV(ObjIdx);
  WRITEV(Offset);
  WRITEV(TgtObjIdx);
  WRITEV(TgtOffset);
}

StorageManager::StorageManager() { std::ios::sync_with_stdio(false); }

Range StorageManager::encodeRange(ObjectManager &OM, uint32_t ObjIdx,
                                  RTObjScheme::TableEntryTy &TE) {
  if (TE.IsNull) {
    return Range(ObjIdx, false, 0, nullptr, nullptr);
  }
  char *ValueP = TE.getBase();
  char *SavedP = TE.SavedValues;
  // If we have started to save values, ensure all are saved.
  char *ShadowP = TE.getShadow();
  uint64_t NoLastPtr = ~0 - sizeof(void *);
  uint64_t LastPtr = NoLastPtr;
  auto ShadowSize = TE.getShadowSize();
  bool AnyRead = TE.AnyRead;
  bool AnyPtrRead = TE.AnyPtrRead;
  for (uint32_t I = 0; (AnyRead || AnyPtrRead) && I < ShadowSize; ++I) {
    unsigned char V = ShadowP[I];
    for (auto Offset : {0, 1}) {
      uint32_t ValueI = 2 * I + Offset;
      if (AnyRead && SavedP && (V & BitsTable[RecordBit][0][Offset]) &&
          !(V & BitsTable[SavedBit][0][Offset]))
        SavedP[ValueI] = ValueP[ValueI];

      bool IsPtr = (V & BitsTable[PtrBit][0][Offset]);
      if (LastPtr + sizeof(void *) == ValueI) {
        char **PtrAddr =
            (char **)(SavedP ? SavedP + LastPtr : ValueP + LastPtr);
        auto [TgtObjIdx, TgtOffset] = OM.getPtrInfo(*PtrAddr, false);
        Ptrs.emplace_back(ObjIdx, LastPtr, TgtObjIdx, TgtOffset);
        LastPtr = NoLastPtr;
      }

      if (IsPtr && LastPtr == NoLastPtr)
        LastPtr = ValueI;
    }
  }
  if (LastPtr + sizeof(void *) == ShadowSize * 2) {
    char **PtrAddr = (char **)(SavedP ? SavedP + LastPtr : ValueP + LastPtr);
    auto [TgtObjIdx, TgtOffset] = OM.getPtrInfo(*PtrAddr, false);
    Ptrs.emplace_back(ObjIdx, LastPtr, TgtObjIdx, TgtOffset);
  }

  char *BaseP = SavedP ? SavedP : ValueP;
  return Range(ObjIdx, AnyRead, TE.getNegativeSize(), BaseP,
               BaseP + TE.getSize());
}

void StorageManager::encode(ObjectManager &OM, uint32_t ObjIdx,
                            RTObjScheme::TableEntryTy &TE) {
  Range R = encodeRange(OM, ObjIdx, TE);
  if (TE.GlobalName) {
    Globals.emplace_back(R, TE.GlobalName);
  } else {
    Ranges.push_back(R);
  }
}

void StorageManager::write(std::ofstream &OFS) {
  DEFINE_WRITEV(OFS);

  WRITEV(FileMagic);

  uint32_t NRanges = Ranges.size();
  WRITEV(NRanges);
  for (auto &Range : Ranges)
    Range.write(OFS);

  uint32_t NGlobals = Globals.size();
  WRITEV(NGlobals);
  for (auto &Global : Globals)
    Global.write(OFS);

  uint32_t NPtrs = Ptrs.size();
  WRITEV(NPtrs);
  for (auto &Ptr : Ptrs)
    Ptr.write(OFS);
}

void StorageManager::read(std::ifstream &IFS, GlobalManager &GM) {
  DEFINE_READV(IFS);

  FileMagicTy Magic;
  READV(Magic);
  if (Magic != FileMagic)
    FATAL();

  uint32_t NRanges;
  READV(NRanges);
  for (uint32_t I = 0; I < NRanges; ++I)
    Ranges.emplace_back(IFS);

  uint32_t NGlobals;
  READV(NGlobals);
  for (uint32_t I = 0; I < NGlobals; ++I)
    Globals.emplace_back(IFS, GM);

  uint32_t NPtrs;
  READV(NPtrs);
  for (uint32_t I = 0; I < NPtrs; ++I)
    Ptrs.emplace_back(IFS);

  for (auto &Ptr : Ptrs) {
    auto &ObjRange = Ranges[Ptr.ObjIdx];
    auto &TgtObjRange = Ranges[Ptr.TgtObjIdx];
    char *ObjPtr = &ObjRange.Begin[Ptr.Offset];
    char *TgtObjPtr =
        &TgtObjRange.Begin[Ptr.TgtOffset + TgtObjRange.NegativeSize];
    *(char **)(ObjPtr) = TgtObjPtr;
    DEBUG("repoint #{} at {} --> #{} at {}\n", Ptr.ObjIdx, (void *)ObjPtr,
          Ptr.TgtObjIdx, (void *)TgtObjPtr);
  }
}

void *StorageManager::getEntryPtr() {
  return Ranges[0].Begin + Ranges[0].NegativeSize;
}

} // namespace __ig::storage

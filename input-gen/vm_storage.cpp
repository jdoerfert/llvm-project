
#include "vm_storage.h"
#include "vm_obj.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>

using namespace __ig;

Range::Range(std::ifstream &IFS) {
  char C;
  IFS >> ObjIdx;
  IFS >> C;
  assert(C == ',');
  IFS >> NegativeSize;
  IFS >> C;
  assert(C == ',');
  IFS >> AnyRecorded;
  IFS >> C;
  assert(C == ',');
  ptrdiff_t Length;
  IFS >> Length;
  IFS >> C;
  assert(C == ',');
  if (Length) {
    Begin = (char *)malloc(Length);
    End = Begin + Length;
    if (AnyRecorded)
      IFS.read(Begin, Length);
  } else {
    Begin = End = nullptr;
  }
  IFS >> C;
  assert(C == ',');
}

void Range::write(std::ofstream &OFS) {
  OFS << ObjIdx << ',' << NegativeSize << ',' << AnyRecorded << ','
      << (End - Begin) << ',';
  if (AnyRecorded)
    OFS.write(Begin, End - Begin);
  OFS << ',';
}

Ptr::Ptr(std::ifstream &IFS) {
  char C;
  IFS >> ObjIdx;
  IFS >> C;
  assert(C == ',');
  IFS >> Offset;
  IFS >> C;
  assert(C == ',');
  IFS >> TgtObjIdx;
  IFS >> C;
  assert(C == ',');
  IFS >> TgtOffset;
  IFS >> C;
  assert(C == ',');
}

void Ptr::write(std::ofstream &OFS) {
  OFS << ObjIdx << ',' << Offset << ',' << TgtObjIdx << ',' << TgtOffset << ',';
}

StorageManager::StorageManager() { std::ios::sync_with_stdio(false); }

void StorageManager::encode(ObjectManager &OM, uint32_t ObjIdx,
                            TableSchemeBaseTy::TableEntryTy &TE) {
  if (TE.IsNull) {
    Ranges.emplace_back(ObjIdx, false, 0, nullptr, nullptr);
    return;
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
  Ranges.emplace_back(ObjIdx, AnyRead, TE.getNegativeSize(), BaseP,
                      BaseP + TE.getSize());
}

void StorageManager::write(std::ofstream &OFS) {
  uint32_t NRanges = Ranges.size();
  OFS << NRanges;
  OFS << ',';
  for (auto &Range : Ranges)
    Range.write(OFS);
  uint32_t NPtrs = Ptrs.size();
  OFS << 'p' << NPtrs << ',';
  for (auto &Ptr : Ptrs)
    Ptr.write(OFS);
}

void *StorageManager::read(std::ifstream &IFS) {
  const int BufferSize = 65536; // Example: 64KB
  char *Buffer = new char[BufferSize];
  IFS.rdbuf()->pubsetbuf(Buffer, BufferSize);
  IFS.tie(nullptr);

  char r;
  uint32_t NRanges;
  IFS >> NRanges;
  IFS >> r;
  assert(r == ',');
  for (uint32_t I = 0; I < NRanges; ++I) {
    Ranges.emplace_back(IFS);
  }
  IFS >> r;
  assert(r == 'p');
  uint32_t NPtrs;
  IFS >> NPtrs;
  IFS >> r;
  assert(r == ',');
  for (uint32_t I = 0; I < NPtrs; ++I) {
    Ptrs.emplace_back(IFS);
  }
  for (auto &Ptr : Ptrs) {
    auto &ObjRange = Ranges[Ptr.ObjIdx];
    auto &TgtObjRange = Ranges[Ptr.TgtObjIdx];
    *(char **)(&ObjRange.Begin[Ptr.Offset]) =
        &TgtObjRange.Begin[Ptr.TgtOffset + TgtObjRange.NegativeSize];
  }
  return Ranges[0].Begin + Ranges[0].NegativeSize;
}

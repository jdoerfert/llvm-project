

#include "vm_obj.h"

#include <cstdint>
#include <fstream>

using namespace __ig;

struct Range {
  uint32_t ObjIdx;
  bool AnyRecorded;
  uint32_t NegativeSize;
  char *Begin, *End;
  Range(uint32_t ObjIdx, bool AnyRecorded, uint32_t NegativeSize, char *Begin,
        char *End)
      : ObjIdx(ObjIdx), AnyRecorded(AnyRecorded), NegativeSize(NegativeSize),
        Begin(Begin), End(End) {}
  Range(std::ifstream &IFS);

  void write(std::ofstream &OFS);
};

struct Ptr {
  uint32_t ObjIdx;
  uint32_t Offset;
  uint32_t TgtObjIdx;
  uint32_t TgtOffset;

  Ptr(uint32_t ObjIdx, uint32_t Offset, uint32_t TgtObjIdx, uint32_t TgtOffset)
      : ObjIdx(ObjIdx), Offset(Offset), TgtObjIdx(TgtObjIdx),
        TgtOffset(TgtOffset) {}
  Ptr(std::ifstream &IFS);

  void write(std::ofstream &OFS);
};

struct StorageManager {
  std::vector<Range> Ranges;
  std::vector<Ptr> Ptrs;

  StorageManager();

  void encode(ObjectManager &OM, uint32_t ObjIdx,
              TableSchemeBaseTy::TableEntryTy &TE);

  void *read(std::ifstream &IFS);
  void write(std::ofstream &OFS);
};

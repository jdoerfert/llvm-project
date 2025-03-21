#include "defer.h"
#include "global_manager.h"
#include "vm_obj.h"

#include <cstdint>
#include <fstream>
#include <string>

using namespace __ig;

namespace __ig::storage {

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
  Range(std::ifstream &IFS, char *Memory);

  void write(std::ofstream &OFS);
};

struct Global {
  DeferLocalConstruction<Range> R;
  std::string Name;
  Global(uint32_t ObjIdx, bool AnyRecorded, uint32_t NegativeSize, char *Begin,
         char *End, std::string Name)
      : Name(Name) {
    R.init(ObjIdx, AnyRecorded, NegativeSize, Begin, End);
  }
  Global(Range R, std::string Name) : Name(Name) { this->R.init(R); }
  Global(std::ifstream &IFS, GlobalManager &GM);

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
  std::vector<Global> Globals;
  std::vector<Range> Ranges;
  std::vector<Ptr> Ptrs;

  StorageManager();

  Range encodeRange(ObjectManager &OM, uint32_t ObjIdx,
                    RTObjScheme::TableEntryTy &TE);

  void encode(ObjectManager &OM, uint32_t ObjIdx,
              RTObjScheme::TableEntryTy &TE);

  void read(std::ifstream &IFS, GlobalManager &GM);
  void write(std::ofstream &OFS);

  void *getEntryPtr();
};

} // namespace __ig::storage

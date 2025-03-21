#include "vm_obj.h"
#include "vm_storage.h"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>

using namespace __ig;
namespace __ig {
extern uint32_t OriginalSeed;
} // namespace __ig

ObjectManager::~ObjectManager() {}

void *ObjectManager::getEntryObj() { return add(8, getRTObjSeed()); }

void ObjectManager::reset() {
  UserObjSmall.reset();
  UserObjLarge.reset();
  RTObjs.reset();
  FVM.reset();
  std::set<char *> ArgMemPtrs;
}

std::function<void(uint32_t)> __ig::ErrorFn;

void ObjectManager::saveInput(uint32_t EntryNo, uint32_t InputIdx,
                              uint32_t ExitCode) {

  ErrorFn = [](uint32_t Code) {
    ERR("Encountered error while saving input with code {}. Aborting.\n", Code);
    exit(1);
  };

  INPUTGEN_DEBUG({
    if (getenv("PRINT_RUNTIME_OBJECTS")) {
      fprintf(stderr, "\n\nRuntime objects (%u):\n", RTObjs.TableEntryCnt);
      for (uint32_t I = 0; I < RTObjs.TableEntryCnt; ++I) {
        RTObjs.Table[I].printStats();
      }
    }
  });

  storage::StorageManager SM;

  for (uint32_t I = 0, E = RTObjs.TableEntryCnt; I != E; ++I) {
    SM.encode(*this, I, RTObjs.Table[I]);
  }

  std::string OutputName = ProgramName + "." + std::to_string(EntryNo) + "." +
                           std::to_string(InputIdx) + "." +
                           std::to_string(ExitCode) + "." +
                           std::to_string(OriginalSeed) + ".inp";
  std::ofstream OFS(OutputName, std::ios_base::out | std::ios_base::binary);
  SM.write(OFS);
}
void __ig::error(uint32_t ErrorCode) {
  if (ErrorFn)
    ErrorFn(ErrorCode);
  else
    printf("Encountered error %u but no stop function available\n", ErrorCode);
  std::terminate();
}

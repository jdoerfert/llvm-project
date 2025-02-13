#include "vm_obj.h"
#include "vm_storage.h"

#include <cstddef>
#include <cstdint>
#include <exception>

using namespace __ig;

void EncodingSchemeTy::error(uint32_t ErrorCode) {
  if (!OM.StopFn) {
    printf("Encountered error %u but no stop function available\n", ErrorCode);
    std::terminate();
  }
  OM.StopFn(ErrorCode);
  std::terminate();
}

ObjectManager::~ObjectManager() {}

void *ObjectManager::getObj(uint32_t Seed) { return add(0, 8, Seed); }

void ObjectManager::reset() {
  UserBS10.reset();
  RTObjs.reset();
  std::set<char *> ArgMemPtrs;
  for (auto &It : BranchConditions)
    for (auto &ItIt : It.second)
      if (ArgMemPtrs.insert(ItIt->ArgMemPtr).second)
        free(ItIt->ArgMemPtr);
  BranchConditions.clear();
}

void ObjectManager::saveInput(uint32_t InputIdx, uint32_t ExitCode) {
#ifndef NDEBUG
  if (getenv("PRINT_RUNTIME_OBJECTS")) {
    printf("\n\nRuntime objects (%u):\n", RTObjs.TableEntryCnt);
    for (uint32_t I = 0; I < RTObjs.TableEntryCnt; ++I) {
      RTObjs.Table[I].printStats();
    }
  }
#endif

  StorageManager SM;

  for (uint32_t I = 0, E = RTObjs.TableEntryCnt; I != E; ++I) {
    SM.encode(*this, I, RTObjs.Table[I]);
  }

  std::string OutputName = ProgramName + "." + std::to_string(InputIdx) + "." +
                           std::to_string(ExitCode) + ".inp";
  std::ofstream OFS(OutputName, std::ios_base::out | std::ios_base::binary);
  SM.write(OFS);
}

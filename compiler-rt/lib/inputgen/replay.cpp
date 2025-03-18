#include "common.h"
#include "global_manager.h"
#include "logging.h"
#include "timer.h"
#include "vm_storage.h"

#include <cstdint>
#include <cstdio>

namespace __ig {
bool GMInit = false;
DeferGlobalConstruction<GlobalManager, GMInit> GM;
} // namespace __ig

using namespace __ig;

int main(int argc, char **argv) {
  if (argc < 2) {
    ERR("Usage: {} <file.inp> [<entry_no>]\n", argv[0]);
    printNumAvailableFunctions();
    printAvailableFunctions();
    exit(static_cast<int>(ExitStatus::WrongUsage));
  }

  uint32_t EntryNo = 0;
  if (argc > 2) 
    EntryNo = std::atoi(argv[2]);
  if (EntryNo >= __ig_num_entry_points) {
    fprintf(stderr, "Entry %u is out of bounds, %u available\n", EntryNo,
            __ig_num_entry_points);
    exit(static_cast<int>(ExitStatus::EntryNoOutOfBounds));
  }

  void *P;
  storage::StorageManager SM;
  {
    Timer T("init");
    std::ifstream IFS(argv[1], std::ios_base::in | std::ios_base::binary);
    const int BufferSize = 65536; // Example: 64KB
    char *Buffer = new char[BufferSize];
    IFS.rdbuf()->pubsetbuf(Buffer, BufferSize);
    IFS.tie(nullptr);

    GM->sort();
#ifndef NDEBUG
    assert(GM.isConstructed());
    std::cerr << "Globals in replay module\n";
    for (auto G : GM->Globals)
      std::cerr << G.Name << "\n";
#endif
    SM.read(IFS, *GM);
#ifndef NDEBUG
    std::cerr << "Globals in input\n";
    for (auto G : SM.Globals)
      std::cerr << G.Name << "\n";
#endif

    P = SM.getEntryPtr();
  }
  {
    Timer T("replay");
    __ig_entry(EntryNo, P);
  }
  exit(static_cast<int>(ExitStatus::Success));
}

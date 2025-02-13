
#include "common.h"
#include "logging.h"
#include "timer.h"
#include "vm_storage.h"

#include <cstdint>
#include <cstdio>

extern "C" char *__ig_entry_point_names[];
extern "C" uint32_t __ig_num_entry_points;
extern "C" void __ig_entry(uint32_t, void *);

int main(int argc, char **argv) {
  if (argc < 2) {
    ERR("Usage: {} <file.inp> [<entry_no>]\n", argv[0]);
    ERR("  Available functions:\n");
    for (uint32_t I = 0; I < __ig_num_entry_points; I++)
      ERR("    {}: {}\n", I, __ig_entry_point_names[I]);
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
  {
    Timer T("init");
    StorageManager SM;
    std::ifstream IFS(argv[1], std::ios_base::in | std::ios_base::binary);
    const int BufferSize = 65536; // Example: 64KB
    char *Buffer = new char[BufferSize];
    IFS.rdbuf()->pubsetbuf(Buffer, BufferSize);
    IFS.tie(nullptr);
    P = SM.read(IFS);
  }
  {
    Timer T("replay");
    __ig_entry(EntryNo, P);
  }
  exit(static_cast<int>(ExitStatus::Success));
}

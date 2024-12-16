
#include "vm_storage.h"
#include "timer.h"

#include <cstdint>
#include <cstdio>

extern "C" uint32_t __ig_num_entry_points;
extern "C" void __ig_entry(uint32_t, void *);

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <file.inp> [<entry_no>]\n", argv[0]);
    exit(1);
  }

  uint32_t EntryNo = 0;
  if (argc > 2) 
    EntryNo = std::atoi(argv[2]);
  if (EntryNo >= __ig_num_entry_points) {
    fprintf(stderr, "Entry %u is out of bounds, %u available\n", EntryNo,
            __ig_num_entry_points);
    exit(1);
  }

  void *P;
  {
    Timer T("init");
    StorageManager SM;
    std::ifstream IFS(argv[1], std::ios_base::in | std::ios_base::binary);
    P = SM.read(IFS);
  }
  {
    Timer T("replay");
    __ig_entry(EntryNo, P);
  }
}


#include "common.h"
#include "logging.h"
#include "timer.h"
#include "vm_storage.h"

#include <cstdint>
#include <cstdio>

#ifndef NDEBUG
#define PRINTF(...) printf(__VA_ARGS__)
#else
#define PRINTF(...)
#endif

extern "C" {
IG_API_ATTRS
void __ig_gen_value(void *pointer, int32_t value_size, int64_t alignment,
                    int32_t value_type_id) {
  PRINTF("load pre -- pointer: %p, value_size: %i, alignment: %lli, "
         "value_type_id: %i\n",
         pointer, value_size, alignment, value_type_id);
  memset(pointer, 0, value_size);
}
}

extern "C" void __ig_entry(uint32_t, void *);

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

#include "include/obj_encoding.h"

namespace __objsan {
using SmallObjectsTy = BucketSchemeTy</*EncodingNo=*/1,
                                      /*OffsetBits=*/12, /*BucketBits=*/3,
                                      /*RealPtrBits=*/32>;
using LargeObjectsTy = LedgerSchemeTy</*EncodingNo=*/2, /*ObjectBits=*/20>;
using FixedObjectsTy =
    FixedLedgerSchemeTy</*EncodingNo=*/3, /*ObjectBits=*/20, 16>;

__attribute__((visibility("default"))) SmallObjectsTy SmallObjects;

__attribute__((visibility("default"))) LargeObjectsTy LargeObjects;

__attribute__((visibility("default"))) FixedObjectsTy FixedObjects;
} // namespace __objsan

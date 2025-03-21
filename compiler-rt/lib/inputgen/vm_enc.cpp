#include "vm_enc.h"
#include "vm_obj.h"

using namespace __ig;

template <uint32_t EncodingNo, uint32_t OffsetBits>
TableSchemeTy<EncodingNo, OffsetBits>::SeedTy
TableSchemeTy<EncodingNo, OffsetBits>::getRTObjSeed() {
  return OM.getRTObjSeed();
}

// Explicit instantiation
template RTObjScheme::SeedTy RTObjScheme::getRTObjSeed();

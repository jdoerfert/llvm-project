add_llvm_library(LLVMFrontend
  OpenMPConstants.cpp
  OpenMPIRBuilder.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Frontend

  DEPENDS
  intrinsics_gen
  )

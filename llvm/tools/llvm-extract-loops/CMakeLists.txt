set(LLVM_LINK_COMPONENTS
  Analysis
  BitWriter
  Core
  IRReader
  IRPrinter
  Passes
  Support
  )

add_llvm_tool(llvm-extract-loops
  llvm-extract-loops.cpp

  DEPENDS
  intrinsics_gen
  )

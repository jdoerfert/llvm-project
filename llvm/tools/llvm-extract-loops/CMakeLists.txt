set(LLVM_LINK_COMPONENTS
  Analysis
  BitWriter
  Core
  IPO
  IRReader
  IRPrinter
  InstCombine
  ScalarOpts
  Passes
  Support
  Target
  TargetParser
  TransformUtils
  Scalar
  Core
  )

add_llvm_tool(llvm-extract-loops
  llvm-extract-loops.cpp

  DEPENDS
  intrinsics_gen
  )

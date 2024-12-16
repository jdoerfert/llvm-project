set(LLVM_LINK_COMPONENTS
  Core
  BitWriter
  IRReader
  Support
  IPO
  Passes
  Analysis
  TransformUtils
  TargetParser
  Instrumentation
)

add_llvm_tool(inputgen-instrument
  inputgen-instrument.cpp
  PARTIAL_SOURCES_INTENDED
  DEPENDS
)

add_llvm_tool(inputgen-mbb-pgo-info
  inputgen-mbb-pgo-info.cpp
  PARTIAL_SOURCES_INTENDED
  DEPENDS
)

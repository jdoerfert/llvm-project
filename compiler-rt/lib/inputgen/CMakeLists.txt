
set(INPUTGEN_CFLAGS
  ${COMPILER_RT_COMMON_CFLAGS}
  -std=c++20
  -flto
  -ffat-lto-objects
)
set(INPUTGEN_LINK_FLAGS
  ${COMPILER_RT_COMMON_LINK_FLAGS}
)

add_custom_target(inputgen.generate)
set_target_properties(inputgen.generate PROPERTIES FOLDER "Compiler-RT Misc")

add_compiler_rt_runtime(inputgen.generate
  STATIC
  ARCHS ${INPUTGEN_SUPPORTED_ARCH}
  SOURCES
    ig.cpp
    vm_obj.cpp
    vm_values.cpp
    vm_storage.cpp
    generate.cpp

  ADDITIONAL_HEADERS ${INPUTGEN_HEADERS}
  CFLAGS ${INPUTGEN_CFLAGS}
  LINK_FLAGS ${INPUTGEN_LINK_FLAGS}
  PARENT_TARGET inputgen_generate)

add_custom_target(inputgen.record)
set_target_properties(inputgen.record PROPERTIES FOLDER "Compiler-RT Misc")

add_compiler_rt_runtime(inputgen.record
  STATIC
  ARCHS ${INPUTGEN_SUPPORTED_ARCH}
  SOURCES
    record.cpp

  ADDITIONAL_HEADERS ${INPUTGEN_HEADERS}
  CFLAGS ${INPUTGEN_CFLAGS}
  LINK_FLAGS ${INPUTGEN_LINK_FLAGS}
  PARENT_TARGET inputgen_record)

add_custom_target(inputgen.replay)
set_target_properties(inputgen.replay PROPERTIES FOLDER "Compiler-RT Misc")

add_compiler_rt_runtime(inputgen.replay
  STATIC
  ARCHS ${INPUTGEN_SUPPORTED_ARCH}
  SOURCES
    vm_storage.cpp
    replay.cpp

  ADDITIONAL_HEADERS ${INPUTGEN_HEADERS}
  CFLAGS ${INPUTGEN_CFLAGS}
  LINK_FLAGS ${INPUTGEN_LINK_FLAGS}
  PARENT_TARGET inputgen_replay)

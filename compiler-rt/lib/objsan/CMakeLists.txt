
set(OBJSAN_CFLAGS
  ${COMPILER_RT_COMMON_CFLAGS}
  -std=c++20
  -fno-exceptions
)
append_rtti_flag(OFF OBJSAN_CFLAGS)

set(OBJSAN_IR_CFLAGS
  ${COMPILER_RT_COMMON_CFLAGS}
  -std=c++20
  -fno-exceptions
  -flto
)

set(OBJSAN_LINK_FLAGS
  ${COMPILER_RT_COMMON_LINK_FLAGS}
)

set(OBJSAN_SOURCES
  objsan_rt.cpp
)
set(OBJSAN_HEADERS
  include/common.h
  include/obj_encoding.h
)
set(OBJSAN_IR_SOURCES
  objsan_ir_rt.cpp
)

add_compiler_rt_component(objsan)

add_custom_target(clang_rt.objsan)
set_target_properties(clang_rt.objsan PROPERTIES FOLDER "Compiler-RT Misc")

add_compiler_rt_runtime(clang_rt.objsan
  STATIC
  OS ${SANITIZER_COMMON_SUPPORTED_OS}
  ARCHS ${OBJSAN_SUPPORTED_ARCH}
  SOURCES ${OBJSAN_SOURCES}

  ADDITIONAL_HEADERS ${OBJSAN_HEADERS}
  CFLAGS ${OBJSAN_CFLAGS}
  LINK_FLAGS ${OBJSAN_LINK_FLAGS}
  PARENT_TARGET objsan)

if(COMPILER_RT_OBJSAN_BUILD_PRELOAD_LIBRARY)
  add_subdirectory(preload)
endif()

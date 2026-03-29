option(OBJSAN_ENABLE_DEBUG "Debug output for objsan" OFF)

set(OBJSAN_CFLAGS
  --std=c++20
  -fno-exceptions
)

append_rtti_flag(OFF OBJSAN_CFLAGS)

if(LLVM_DEFAULT_TARGET_TRIPLE)
  list(APPEND OBJSAN_CFLAGS --target=${LLVM_DEFAULT_TARGET_TRIPLE})
endif()

# The host objsan rt lib type must be either STATIC or SHARED, because in the
# clang toolchain arg generation only those types are supported.
set(OBJSAN_RT_LIB_TYPE STATIC)

if("${LLVM_DEFAULT_TARGET_TRIPLE}" MATCHES "^amdgcn" OR
   "${CMAKE_CXX_COMPILER_TARGET}" MATCHES "^amdgcn")
  #list(APPEND OBJSAN_CFLAGS "SHELL:-Xclang -mcode-object-version=none")
  list(APPEND OBJSAN_CFLAGS "-D__OBJSAN_DEVICE__")
  list(APPEND OBJSAN_CFLAGS -nogpulib)
  list(APPEND OBJSAN_CFLAGS -nostdlibinc)
  list(APPEND OBJSAN_CFLAGS -fno-rtti)
  list(APPEND OBJSAN_CFLAGS -fno-exceptions)
  list(APPEND OBJSAN_CFLAGS -fconvergent-functions)
  list(APPEND OBJSAN_CFLAGS "-D__OBJSAN_USE_START_STOP_SECTION_CTOR__")
  set(OBJSAN_RT_LIB_TYPE OBJECT)
elseif("${LLVM_DEFAULT_TARGET_TRIPLE}" MATCHES "^nvptx" OR
       "${CMAKE_CXX_COMPILER_TARGET}" MATCHES "^nvptx")
  list(APPEND OBJSAN_CFLAGS "-D__OBJSAN_DEVICE__")
  # We need LTO because otherwise we need to specify the GPU architecture
  # (-march=sm_XX) and we want this to be reusable across many archs
  list(APPEND OBJSAN_CFLAGS -flto)
  list(APPEND OBJSAN_CFLAGS -nogpulib)
  list(APPEND OBJSAN_CFLAGS -nostdlibinc)
  list(APPEND OBJSAN_CFLAGS -fno-rtti)
  list(APPEND OBJSAN_CFLAGS -fno-exceptions)
  list(APPEND OBJSAN_CFLAGS -fconvergent-functions)
  list(APPEND OBJSAN_CFLAGS -Wno-unknown-cuda-version)
  # clang-nvlink-wrapper does not properly link in static libraries (.a files)
  # made up of LLVM IR (-flto), so we opt for OBJECT
  set(OBJSAN_RT_LIB_TYPE OBJECT)
else()
  list(APPEND OBJSAN_CFLAGS "-D__OBJSAN_USE_START_STOP_SECTION_CTOR__")
  add_subdirectory(preload)
endif()

if(OBJSAN_ENABLE_DEBUG)
  list(APPEND OBJSAN_CFLAGS "-DOBJSAN_DEBUG")
endif()

set(OBJSAN_IR_CFLAGS
  ${OBJSAN_CFLAGS}
  -flto
)

list(APPEND OBJSAN_CFLAGS ${COMPILER_RT_COMMON_CFLAGS})

set(OBJSAN_LINK_FLAGS
  ${COMPILER_RT_COMMON_LINK_FLAGS}
)
set(OBJSAN_IR_LINK_FLAGS
  ${OBJSAN_LINK_FLAGS}
)

set(OBJSAN_SOURCES
  objsan_rt.cpp
)
set(OBJSAN_HEADERS
  include/common.h
  include/obj_encoding.h
)

add_compiler_rt_component(objsan)
add_compiler_rt_component(objsan-ir)

add_custom_target(clang_rt.objsan)
add_custom_target(clang_rt.objsan-ir)
set_target_properties(clang_rt.objsan PROPERTIES FOLDER "Compiler-RT Misc")
set_target_properties(clang_rt.objsan-ir PROPERTIES FOLDER "Compiler-RT Misc")

add_compiler_rt_runtime(clang_rt.objsan
  ${OBJSAN_RT_LIB_TYPE}
  OS ${SANITIZER_COMMON_SUPPORTED_OS}
  ARCHS ${OBJSAN_SUPPORTED_ARCH}
  SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/objsan_rt.cpp

  ADDITIONAL_HEADERS ${OBJSAN_HEADERS}
  CFLAGS ${OBJSAN_CFLAGS}
  LINK_FLAGS ${OBJSAN_LINK_FLAGS}
  PARENT_TARGET objsan)

add_compiler_rt_runtime(clang_rt.objsan-ir
  OBJECT
  OS ${SANITIZER_COMMON_SUPPORTED_OS}
  ARCHS ${OBJSAN_SUPPORTED_ARCH}
  SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/objsan_ir_rt.cpp

  ADDITIONAL_HEADERS ${OBJSAN_HEADERS}
  CFLAGS ${OBJSAN_IR_CFLAGS}
  PARENT_TARGET objsan)

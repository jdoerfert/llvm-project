set(OBJSAN_CFLAGS
  ${COMPILER_RT_COMMON_CFLAGS}
  -std=c++20
  -fno-exceptions
)
append_rtti_flag(OFF OBJSAN_CFLAGS)


if(LLVM_DEFAULT_TARGET_TRIPLE)
  list(APPEND OBJSAN_CFLAGS --target=${LLVM_DEFAULT_TARGET_TRIPLE})
endif()

if("${LLVM_DEFAULT_TARGET_TRIPLE}" MATCHES "^amdgcn" OR
   "${CMAKE_CXX_COMPILER_TARGET}" MATCHES "^amdgcn")
  set(target_name "amdgpu")
  list(APPEND OBJSAN_CFLAGS "SHELL:-Xclang -mcode-object-version=none")
  list(APPEND OBJSAN_CFLAGS "-D__OBJSAN_DEVICE__")
  list(APPEND OBJSAN_CFLAGS "SHELL: -mllvm -vectorize-slp=false")
  list(APPEND OBJSAN_CFLAGS -flto)
  list(APPEND OBJSAN_CFLAGS -nogpulib)
  list(APPEND OBJSAN_CFLAGS -nostdlibinc)
  list(APPEND OBJSAN_CFLAGS -fno-rtti)
  list(APPEND OBJSAN_CFLAGS -fno-exceptions)
  list(APPEND OBJSAN_CFLAGS -fconvergent-functions)
elseif("${LLVM_DEFAULT_TARGET_TRIPLE}" MATCHES "^nvptx" OR
       "${CMAKE_CXX_COMPILER_TARGET}" MATCHES "^nvptx")
  set(target_name "nvptx")
  list(APPEND OBJSAN_CFLAGS --cuda-feature=+ptx63)
  list(APPEND OBJSAN_CFLAGS "-D__OBJSAN_DEVICE__")
  list(APPEND OBJSAN_CFLAGS "SHELL: -mllvm -vectorize-slp=false")
  list(APPEND OBJSAN_CFLAGS -flto)
  list(APPEND OBJSAN_CFLAGS -nogpulib)
  list(APPEND OBJSAN_CFLAGS -nostdlibinc)
  list(APPEND OBJSAN_CFLAGS -fno-rtti)
  list(APPEND OBJSAN_CFLAGS -fno-exceptions)
  list(APPEND OBJSAN_CFLAGS -fconvergent-functions)
  list(APPEND OBJSAN_CFLAGS -Wno-unknown-cuda-version)
else()
  add_subdirectory(preload)
endif()

set(OBJSAN_IR_CFLAGS
  ${OBJSAN_CFLAGS}
  -flto
)

set(OBJSAN_LINK_FLAGS
  ${COMPILER_RT_COMMON_LINK_FLAGS}
)
set(OBJSAN_IR_LINK_FLAGS
  ${OBJSAN_LINK_FLAGS}
)

set(OBJSAN_SOURCES
  objsan_rt.cpp
)
set(OBJSAN_IR_SOURCES
  objsan_ir_rt.cpp
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
  STATIC
  OS ${SANITIZER_COMMON_SUPPORTED_OS}
  ARCHS ${OBJSAN_SUPPORTED_ARCH}
  SOURCES ${OBJSAN_SOURCES}

  ADDITIONAL_HEADERS ${OBJSAN_HEADERS}
  CFLAGS ${OBJSAN_CFLAGS}
  LINK_FLAGS ${OBJSAN_LINK_FLAGS}
  PARENT_TARGET objsan)

add_compiler_rt_runtime(clang_rt.objsan-ir
  STATIC
  OS ${SANITIZER_COMMON_SUPPORTED_OS}
  ARCHS ${OBJSAN_SUPPORTED_ARCH}
  SOURCES ${OBJSAN_IR_SOURCES}

  ADDITIONAL_HEADERS ${OBJSAN_HEADERS}
  CFLAGS ${OBJSAN_IR_CFLAGS}
  LINK_FLAGS ${OBJSAN_IR_LINK_FLAGS}
  PARENT_TARGET objsan)

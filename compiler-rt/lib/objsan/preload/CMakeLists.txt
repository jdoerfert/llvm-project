string(TOLOWER "${COMPILER_RT_OBJSAN_PRELOAD_LIBS_TO_BUILD}" COMPILER_RT_OBJSAN_PRELOAD_LIBS_TO_BUILD_LOWER)

set(src_files_common objsan_preload.cpp)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../include")

if ("cuda" IN_LIST COMPILER_RT_OBJSAN_PRELOAD_LIBS_TO_BUILD_LOWER)
  message("Building objsan cuda preload")
  find_package(CUDA REQUIRED)
  set(OBJSAN_PRELOAD_CUDA_SOURCES
    ${src_files_common}
    objsan_preload_cuda.cpp
    objsan_preload_impl_cuda.cpp
  )
  set(OBJSAN_PRELOAD_CUDA_CFLAGS
    -I${CUDA_INCLUDE_DIRS}
    -x cuda
    -fgpu-rdc
    -foffload-lto
  )
  set(OBJSAN_PRELOAD_CUDA_LINK_FLAGS
    -lcudart -L${CUDA_TOOLKIT_ROOT_DIR}/lib64)
  add_compiler_rt_runtime(clang_rt.objsan_preload_cuda
    SHARED
    OS ${SANITIZER_COMMON_SUPPORTED_OS}
    ARCHS ${OBJSAN_SUPPORTED_ARCH}
    SOURCES ${OBJSAN_PRELOAD_CUDA_SOURCES}
    CFLAGS ${OBJSAN_PRELOAD_CUDA_CFLAGS}
    LINK_FLAGS ${OBJSAN_PRELOAD_CUDA_LINK_FLAGS}
    PARENT_TARGET objsan_preload_cuda)
endif()

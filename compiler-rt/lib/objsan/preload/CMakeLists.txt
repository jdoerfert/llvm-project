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
  # TODO the way we specify the -I needs to be fixed - I think the variable can
  # in theory contain multiple paths
  # TODO sm_80
  set(OBJSAN_PRELOAD_CUDA_CFLAGS
    -I${CUDA_INCLUDE_DIRS}
    -x cuda
    -fgpu-rdc
    -foffload-lto
    --offload-link
    --offload-arch=sm_90
  )
  if(OBJSAN_ENABLE_DEBUG)
    list(APPEND OBJSAN_PRELOAD_CUDA_CFLAGS "-DOBJSAN_DEBUG")
  endif()
  # TODO better way to get the library path?
  set(OBJSAN_PRELOAD_CUDA_LINK_FLAGS
    -lcudart -L${CUDA_TOOLKIT_ROOT_DIR}/lib64)
  add_compiler_rt_runtime(clang_rt.objsan_preload_cuda
    STATIC
    OS ${SANITIZER_COMMON_SUPPORTED_OS}
    ARCHS ${OBJSAN_SUPPORTED_ARCH}
    SOURCES ${OBJSAN_PRELOAD_CUDA_SOURCES}
    CFLAGS ${OBJSAN_PRELOAD_CUDA_CFLAGS}
    LINK_FLAGS ${OBJSAN_PRELOAD_CUDA_LINK_FLAGS}
    PARENT_TARGET objsan_preload_cuda)
endif()

if ("hip" IN_LIST COMPILER_RT_OBJSAN_PRELOAD_LIBS_TO_BUILD_LOWER)
  message("Building objsan hip preload")
  find_package(HIP REQUIRED)
  set(OBJSAN_PRELOAD_HIP_SOURCES
    ${src_files_common}
    objsan_preload_hip.cpp
    objsan_preload_impl_hip.cpp
  )
  # TODO gfx942
  set(OBJSAN_PRELOAD_HIP_CFLAGS
    -I${HIP_INCLUDE_DIRS}
    -x hip
    -fgpu-rdc
    -foffload-lto
    --offload-link
    --offload-arch=gfx942
    --offload-new-driver
  )
  if(OBJSAN_ENABLE_DEBUG)
    list(APPEND OBJSAN_PRELOAD_HIP_CFLAGS "-DOBJSAN_DEBUG")
  endif()
  # TODO better way to get the library path?
  set(OBJSAN_PRELOAD_HIP_LINK_FLAGS
    -lamdhip64 -L${HIP_LIB_INSTALL_DIR})
  add_compiler_rt_runtime(clang_rt.objsan_preload_hip
    STATIC
    OS ${SANITIZER_COMMON_SUPPORTED_OS}
    ARCHS ${OBJSAN_SUPPORTED_ARCH}
    SOURCES ${OBJSAN_PRELOAD_HIP_SOURCES}
    CFLAGS ${OBJSAN_PRELOAD_HIP_CFLAGS}
    PARENT_TARGET objsan_preload_hip)
endif()

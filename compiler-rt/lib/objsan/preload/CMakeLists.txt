string(TOLOWER ${COMPILER_RT_OBJSAN_PRELOAD_LIBS_TO_BUILD} LIBS_TO_BUILD)

set(src_files_common objsan_preload.cpp)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../include")

if ("cuda" IN_LIST LIBS_TO_BUILD)
  find_package(CUDA REQUIRED)
  set(src_files ${src_files_common}
                objsan_preload_cuda.cpp
                # objsan_preload_impl_cuda.cu
  )
  add_library(objsan_preload_cuda SHARED ${src_files})
  target_include_directories(objsan_preload_cuda PRIVATE ${CUDA_INCLUDE_DIRS})
endif()

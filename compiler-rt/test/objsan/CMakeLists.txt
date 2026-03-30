
set(OBJSAN_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS})

set(OBJSAN_TESTSUITES)

if (COMPILER_RT_BUILD_OBJSAN)
  set(OBJSAN_PRELOAD_CUDA_LIB_PATH "${COMPILER_RT_LIBRARY_OUTPUT_DIR}/libclang_rt.objsan_preload_cuda.so")
  set(OBJSAN_PRELOAD_HIP_LIB_PATH "${COMPILER_RT_LIBRARY_OUTPUT_DIR}/libclang_rt.objsan_preload_hip.so")
  if(NOT CLANG_RESOURCE_DIR)
    set(CLANG_RESOURCE_DIR "lib/clang/${LLVM_VERSION_MAJOR}")
  endif()
  if(NOT LLVM_ENABLE_PER_TARGET_RUNTIME_DIR)
    error("Need LLVM_ENABLE_PER_TARGET_RUNTIME_DIR")
  endif()
  set(OBJSAN_PRELOAD_CUDA_LIB_PATH "${LLVM_BINARY_DIR}/${CLANG_RESOURCE_DIR}/lib/${LLVM_DEFAULT_TARGET_TRIPLE}/libclang_rt.objsan_preload_cuda.a")
  set(OBJSAN_RUNTIME_CUDA_LIB_PATH "${LLVM_BINARY_DIR}/${CLANG_RESOURCE_DIR}/lib/nvptx64-nvidia-cuda/clang_rt.objsan.o")
  message("Using preload objsan cuda lib ${OBJSAN_PRELOAD_CUDA_LIB_PATH}")
  set(OBJSAN_PRELOAD_HIP_LIB_PATH "${LLVM_BINARY_DIR}/${CLANG_RESOURCE_DIR}/lib/${LLVM_DEFAULT_TARGET_TRIPLE}/libclang_rt.objsan_preload_hip.a")
  set(OBJSAN_RUNTIME_HIP_LIB_PATH "${LLVM_BINARY_DIR}/${CLANG_RESOURCE_DIR}/lib/amdgcn-amd-amdhsa/clang_rt.objsan.o")
  message("Using preload objsan hip lib ${OBJSAN_PRELOAD_HIP_LIB_PATH}")

  find_package(CUDA)
  find_package(HIP)

  set(CONFIG_NAME ${LLVM_DEFAULT_TARGET_TRIPLE})
  list(APPEND OBJSAN_TEST_DEPS objsan)
  configure_lit_site_cfg(
      ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
      ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg.py)
  list(APPEND OBJSAN_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})
endif()

add_lit_testsuite(check-objsan "Running the objsan runtime tests"
  ${OBJSAN_TESTSUITES}
  DEPENDS ${OBJSAN_TEST_DEPS})

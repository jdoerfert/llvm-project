message(STATUS "Building offloading runtime library libomptarget.")

if(LIBOMP_STANDALONE)
  set(LIBOMP ${LIBOMP_STANDALONE})
else()
  set(LIBOMP omp)
endif()

add_llvm_library(omptarget
  SHARED

  device.cpp
  interface.cpp
  omptarget.cpp
  OffloadRTL.cpp
  LegacyAPI.cpp
  PluginManager.cpp
  DeviceImage.cpp

  OpenMP/API.cpp
  OpenMP/Mapping.cpp
  OpenMP/InteropAPI.cpp
  OpenMP/OMPT/Callback.cpp

  KernelLanguage/API.cpp

  ADDITIONAL_HEADER_DIRS
  ${LIBOMPTARGET_INCLUDE_DIR}
  ${LIBOMPTARGET_BINARY_INCLUDE_DIR}

  LINK_COMPONENTS
  FrontendOpenMP
  Support
  Object

  LINK_LIBS
  PUBLIC
  ${LIBOMP}

  NO_INSTALL_RPATH
  BUILDTREE_ONLY
)
target_include_directories(omptarget PRIVATE
  ${LIBOMPTARGET_INCLUDE_DIR} ${LIBOMPTARGET_BINARY_INCLUDE_DIR}
)

if (LIBOMP_HAVE_VERSION_SCRIPT_FLAG)
  target_link_libraries(omptarget PRIVATE
    "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports")
endif()

# Define the TARGET_NAME and DEBUG_PREFIX.
target_compile_definitions(omptarget PRIVATE
  TARGET_NAME=omptarget
  DEBUG_PREFIX="omptarget"
)

foreach(plugin IN LISTS LIBOMPTARGET_PLUGINS_TO_BUILD)
  target_link_libraries(omptarget PRIVATE omptarget.rtl.${plugin})
endforeach()

target_compile_options(omptarget PRIVATE ${offload_compile_flags})
target_link_options(omptarget PRIVATE ${offload_link_flags})

add_llvm_library(offload.kernels
  STATIC

  Kernels/Sanitizer.cpp
  
  LINK_LIBS
  PUBLIC
  omptarget.devicertl

  NO_INSTALL_RPATH
  BUILDTREE_ONLY
)


set(all_amdgpu_architectures "gfx700;gfx701;gfx801;gfx803;gfx900;gfx902;gfx906"
                             "gfx908;gfx90a;gfx90c;gfx940;gfx941;gfx942;gfx1010"
                             "gfx1030;gfx1031;gfx1032;gfx1033;gfx1034;gfx1035"
                             "gfx1036;gfx1100;gfx1101;gfx1102;gfx1103;gfx1150"
                             "gfx1151;gfx1152")
set(all_nvptx_architectures "sm_35;sm_37;sm_50;sm_52;sm_53;sm_60;sm_61;sm_62"
                            "sm_70;sm_72;sm_75;sm_80;sm_86;sm_87;sm_89;sm_90")
set(all_gpu_architectures
    "${all_amdgpu_architectures};${all_nvptx_architectures}")

set(LIBOMPTARGET_DEVICE_ARCHITECTURES "all" CACHE STRING
    "List of device architectures to be used to compile the OpenMP DeviceRTL.")

if(LIBOMPTARGET_DEVICE_ARCHITECTURES STREQUAL "all")
  set(LIBOMPTARGET_DEVICE_ARCHITECTURES ${all_gpu_architectures})
elseif(LIBOMPTARGET_DEVICE_ARCHITECTURES STREQUAL "auto" OR
       LIBOMPTARGET_DEVICE_ARCHITECTURES STREQUAL "native")
  if(NOT LIBOMPTARGET_NVPTX_ARCH AND NOT LIBOMPTARGET_AMDGPU_ARCH)
    message(FATAL_ERROR
      "Could not find 'amdgpu-arch' and 'nvptx-arch' tools required for 'auto'")
  elseif(NOT LIBOMPTARGET_FOUND_NVIDIA_GPU AND NOT LIBOMPTARGET_FOUND_AMDGPU_GPU)
    message(FATAL_ERROR "No AMD or NVIDIA GPU found on the system when using 'auto'")
  endif()
  set(LIBOMPTARGET_DEVICE_ARCHITECTURES
      "${LIBOMPTARGET_NVPTX_DETECTED_ARCH_LIST};${LIBOMPTARGET_AMDGPU_DETECTED_ARCH_LIST}")
endif()
list(REMOVE_DUPLICATES LIBOMPTARGET_DEVICE_ARCHITECTURES)

list(JOIN LIBOMPTARGET_DEVICE_ARCHITECTURES "," KERNEL_OFFLOAD_ARCHS)
target_compile_options(offload.kernels PRIVATE -x cuda --offload-arch=${KERNEL_OFFLOAD_ARCHS} -nocudalib -nogpulib -foffload-lto -foffload-via-llvm )
target_link_options(offload.kernels PRIVATE -x cuda --offload-arch=${KERNEL_OFFLOAD_ARCHS} -nocudalib -nogpulib -foffload-lto -foffload-via-llvm )

# libomptarget.so needs to be aware of where the plugins live as they
# are now separated in the build directory.
set_target_properties(omptarget PROPERTIES
                      POSITION_INDEPENDENT_CODE ON
                      INSTALL_RPATH "$ORIGIN"
                      BUILD_RPATH "$ORIGIN:${CMAKE_CURRENT_BINARY_DIR}/..")
install(TARGETS omptarget LIBRARY COMPONENT omptarget DESTINATION "${OFFLOAD_INSTALL_LIBDIR}")


set(OBJSAN_CFLAGS
  ${COMPILER_RT_COMMON_CFLAGS}
  -std=c++20
  -fno-exceptions
  -flto
)

append_rtti_flag(OFF OBJSAN_CFLAGS)

set(OBJSAN_LINK_FLAGS
  ${COMPILER_RT_COMMON_LINK_FLAGS}
)

set(OBJSAN_SOURCES
  objsan_rt.cpp
)
set(OBJSAN_HEADERS
  include/obj_encoding.h
)

add_compiler_rt_component(objsan)

add_custom_target(clang_rt.objsan)
set_target_properties(clang_rt.objsan PROPERTIES FOLDER "Compiler-RT Misc")

foreach(arch ${OBJSAN_SUPPORTED_ARCH})
  foreach(os ${SANITIZER_COMMON_SUPPORTED_OS})
    add_compiler_rt_runtime(clang_rt.objsan-${arch}
      STATIC
      OS ${os}
      ARCHS ${arch}
      SOURCES ${OBJSAN_SOURCES}

      ADDITIONAL_HEADERS ${ASAN_HEADERS}
      CFLAGS ${OBJSAN_CFLAGS}
      LINK_FLAGS ${OBJSAN_LINK_FLAGS}
      PARENT_TARGET objsan)
  endforeach()
endforeach()

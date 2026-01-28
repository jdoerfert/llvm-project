
set(OBJSAN_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS})

set(OBJSAN_TEST_ARCH ${OBJSAN_SUPPORTED_ARCH})

list(REMOVE_ITEM OBJSAN_TEST_ARCH nvptx64)
list(REMOVE_ITEM OBJSAN_TEST_ARCH amdgcn)

set(OBJSAN_TESTSUITES)
set(OBJSAN_DYNAMIC_TESTSUITES)

if (COMPILER_RT_BUILD_OBJSAN)
  message("Adding objsan tests for arches ${OBJSAN_TEST_ARCH}")
  foreach(arch ${OBJSAN_TEST_ARCH})
    string(TOUPPER ${arch} ARCH_UPPER_CASE)
    set(CONFIG_NAME ${ARCH_UPPER_CASE}${OS_NAME}Config)
    list(APPEND OBJSAN_TEST_DEPS objsan)
    configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg.py)
    list(APPEND OBJSAN_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})
  endforeach()
endif()

add_lit_testsuite(check-objsan "Running the objsan runtime tests"
  ${OBJSAN_TESTSUITES}
  DEPENDS ${OBJSAN_TEST_DEPS})

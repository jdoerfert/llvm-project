
set(OBJSAN_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS})

set(OBJSAN_TEST_ARCH ${OBJSAN_SUPPORTED_ARCH})
if (COMPILER_RT_BUILD_OBJSAN)
  list(APPEND OBJSAN_TEST_DEPS objsan)
  configure_lit_site_cfg(
      ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
      ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py)
  list(APPEND OBJSAN_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR})
endif()

add_lit_testsuite(check-objsan-rt "Running the objsan runtime tests"
  ${OBJSAN_TESTSUITES}
  DEPENDS ${OBJSAN_TEST_DEPS})

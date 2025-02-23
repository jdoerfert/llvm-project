
set(INPUTGEN_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS})

set(INPUTGEN_TEST_ARCH ${INPUTGEN_SUPPORTED_ARCH})
if (COMPILER_RT_BUILD_INPUTGEN)
  list(APPEND INPUTGEN_TEST_DEPS inputgen.record inputgen.generate inputgen.replay)
  configure_lit_site_cfg(
      ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
      ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py)
  list(APPEND INPUTGEN_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR})
endif()

add_lit_testsuite(check-inputgen-rt "Running the input-gen runtime tests"
  ${INPUTGEN_TESTSUITES}
  DEPENDS ${INPUTGEN_TEST_DEPS})


set(FLANG_F2CPP_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(FLANG_F2CPP_TEST_DEPS
  flang2cpp
  f2cpp_rt
)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

add_lit_testsuite(check-flang-f2cpp "Running the f2cpp regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${FLANG_F2CPP_TEST_DEPS}
)

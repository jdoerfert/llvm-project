add_llvm_example_library(flang2cpp
    MODULE
    F2CPP.cpp
    PARTIAL_SOURCES_INTENDED

    DEPENDS
    acc_gen
    flangFrontend
    omp_gen
)

add_library(f2cpp_rt STATIC
    f2cpp_rt.cpp
)

target_include_directories(f2cpp_rt PUBLIC include)

add_subdirectory(tests)

cmake_minimum_required(VERSION 3.21)
project(cadfs LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
# -O3 for search benchmarks; add -march=native locally if binaries need not be portable
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

option(CADFS_BUILD_PYTHON "Build pybind11 module" ON)
option(CADFS_BUILD_TESTS  "Build unit tests" ON)
option(CADFS_NATIVE_ARCH "Enable -march=native for local benchmarks" OFF)
option(CADFS_ENABLE_IPO "Enable supported interprocedural optimization" ON)

# ---- Core search library (no Python dependency) ----
add_library(cadfs_core STATIC
        cpp/src/grid_map.cpp
        cpp/src/heuristics.cpp
        cpp/src/risk.cpp
        cpp/src/mlp.cpp
        cpp/src/fast_mlp.cpp
        cpp/src/expert.cpp
        cpp/src/confidence.cpp
        cpp/src/controller.cpp
        cpp/src/search_astar.cpp
        cpp/src/search_focal.cpp
        cpp/src/search_cadfs.cpp
        cpp/src/search_cadfs_next.cpp
        cpp/src/instance_io.cpp
)
target_include_directories(cadfs_core PUBLIC cpp/include)
set_target_properties(cadfs_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(CADFS_NATIVE_ARCH AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cadfs_core PRIVATE -march=native)
endif()
if(CADFS_ENABLE_IPO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT CADFS_IPO_SUPPORTED)
    if(CADFS_IPO_SUPPORTED)
        set_property(TARGET cadfs_core PROPERTY
                     INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    else()
        message(STATUS "CADFS IPO unavailable")
    endif()
endif()

# ---- Python bindings ----
if(CADFS_BUILD_PYTHON)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(cadfs_engine cpp/bindings/bindings.cpp)
    target_link_libraries(cadfs_engine PRIVATE cadfs_core)
    if(CADFS_ENABLE_IPO AND CADFS_IPO_SUPPORTED)
        set_property(TARGET cadfs_engine PROPERTY
                     INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endif()
endif()

# ---- Tests ----
if(CADFS_BUILD_TESTS)
    add_executable(cadfs_tests cpp/tests/test_main.cpp)
    target_link_libraries(cadfs_tests PRIVATE cadfs_core)
    if(CADFS_ENABLE_IPO AND CADFS_IPO_SUPPORTED)
        set_property(TARGET cadfs_tests PROPERTY
                     INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endif()
    include(CTest)
    add_test(NAME cadfs_tests COMMAND cadfs_tests)
endif()

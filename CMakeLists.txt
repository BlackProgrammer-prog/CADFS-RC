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

# ---- Core search library (no Python dependency) ----
add_library(cadfs_core STATIC
        cpp/src/grid_map.cpp
        cpp/src/heuristics.cpp
        cpp/src/risk.cpp
        cpp/src/mlp.cpp
        cpp/src/search_astar.cpp
        cpp/src/search_focal.cpp
        cpp/src/search_cadfs.cpp
        cpp/src/instance_io.cpp
)
target_include_directories(cadfs_core PUBLIC cpp/include)
set_target_properties(cadfs_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---- Python bindings ----
if(CADFS_BUILD_PYTHON)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(cadfs_engine cpp/bindings/bindings.cpp)
    target_link_libraries(cadfs_engine PRIVATE cadfs_core)
endif()

# ---- Tests ----
if(CADFS_BUILD_TESTS)
    add_executable(cadfs_tests cpp/tests/test_main.cpp)
    target_link_libraries(cadfs_tests PRIVATE cadfs_core)
    include(CTest)
    add_test(NAME cadfs_tests COMMAND cadfs_tests)
endif()

# cpp/CMakeLists/txt

# CMakeList is not required during runtime, nor for rebuild of binaries (.pyd files)
# left only for IDE integration


cmake_minimum_required(VERSION 3.31)
project(breachers)

set(CMAKE_CXX_STANDARD 20)
set(PYBIND11_PYTHON_VERSION "3.12")
set(PYBIND11_FINDPYTHON ON)




set(SOURCES_ANT_COLONY ant_colony_source/ant_colony.cpp)
set(SOURCES_BREACH_SOLVER breach_solver_source/breach_solver.cpp)


add_subdirectory(pybind11)
# Not included, requre installation by running in terminal:
# git clone https://github.com/pybind/pybind11.git cpp/pybind11


pybind11_add_module(antcol_cpp ${SOURCES_ANT_COLONY})
pybind11_add_module(bruter_cpp ${SOURCES_BREACH_SOLVER})


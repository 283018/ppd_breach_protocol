# CMakeList is not required during runtime, nor for rebuild of binaries (.pyd files)
# left only for correct inspection of .cpp files


cmake_minimum_required(VERSION 3.31)
project(breachers)

set(CMAKE_CXX_STANDARD 20)




add_subdirectory(pybind11)
# not included, requre installation by running in terminal:
# cd .\cpp\
# git clone https://github.com/pybind/pybind11.git
# cd..

pybind11_add_module(antcol_cpp ant_colony_source/ant_colony.cpp)
pybind11_add_module(bruter_cpp breach_solver_source/breach_solver.cpp)


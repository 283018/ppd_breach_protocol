# breach_solvers/solvers/brute/breach_solver_back/setup.py
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import pybind11


# dir_name = os.path.dirname(__file__)
dir_name = os.path.abspath(os.path.dirname(__file__))

if sys.platform == "win32":
    extra_compile_args = ["/std:c++20", "/openmp", "/O2"]
    extra_link_args = []
else:
    extra_compile_args = ["-std=c++20", "-fopenmp", "-O3"]
    extra_link_args = ["-fopenmp"]

ext_modules = [
    Extension(
        "breach_solver_cpp",
        # ["breach_solver/breach_solver.cpp"],
        [os.path.join(dir_name, "breach_solver", "breach_solver.cpp")],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        "ant_colony_cpp",
        # ["ant_colony\\ant_colony.cpp"],
        [os.path.join(dir_name, "ant_colony", "ant_colony.cpp")],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

class PlatformBuildExt(build_ext):
    def get_ext_fullpath(self, ext_name):
        if sys.platform == "win32":
            plat_subdir = "win"
        elif sys.platform == "darwin":
            plat_subdir = "mac"
        else:
            plat_subdir = sys.platform.lower()

        filename = self.get_ext_filename(ext_name)

        setup_dir = os.path.dirname(os.path.abspath(__file__))
        target_dir = os.path.join(setup_dir, "build", plat_subdir)

        os.makedirs(target_dir, exist_ok=True)

        return os.path.join(target_dir, filename)

setup(
    name="solver_cpp_extensions",
    version="0.1",
    ext_modules=ext_modules,
    cmdclass={"build_ext": PlatformBuildExt},
    zip_safe=False,
)


r'''
To rebuild run:
python .\cpp\setup.py build clean --all
or
python3 ./cpp/setup.py build clean --all
'''
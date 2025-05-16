# breach_solvers/__init__.py

import os
import platform
import sys
import importlib.util

system = platform.system()
if system == "Windows":
    build_subdir = "win"
elif system == "Darwin":
    build_subdir = "mac"
else:
    build_subdir = "linux"

package_dir = os.path.dirname(__file__)
build_dir = os.path.join(package_dir, "build", build_subdir)

module_name = "breach_solver_cpp"
module_path = None

try:
    for file in os.listdir(build_dir):
        if file.startswith(module_name) and (file.endswith(".pyd") or file.endswith(".so") or file.endswith(".dylib")):
            module_path = os.path.join(build_dir, file)
            break
except FileNotFoundError:
    raise ImportError(
        "Build directory not found. Please build the module first:\n"
        r"'python .\breach_solvers\brute\breach_solver_back\setup.py build clean --all'"
        "&&"
        "'python3 ./breach_solvers/brute/breach_solver_back/setup.py build clean --all'"
    )

if module_path is None:
    if not os.path.exists(build_dir):
        raise ImportError(
            "Build directory not found. Please build the module first:\n"
            r"'python .\breach_solvers\brute\breach_solver_back\setup.py build clean --all'"
            "&&"
            "'python3 ./breach_solvers/brute/breach_solver_back/setup.py build clean --all'"
        )
    else:
        raise ImportError(
            f"Compiled module '{module_name}' not found in {build_dir}. Please build the module:\n"
            r"'python .\breach_solvers\brute\breach_solver_back\setup.py build clean --all'"
            "&&"
            "'python3 ./breach_solvers/brute/breach_solver_back/setup.py build clean --all'"
        )


spec = importlib.util.spec_from_file_location(module_name, module_path)
if spec is None:
    raise ImportError(f"Failed to load module '{module_name}' from {module_path}")

module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)


try:
    solve_breach = module.solve_breach
    __all__ = ['solve_breach']
except AttributeError:
    raise ImportError(f"Module '{module_name}' does not contain 'solve_breach' function")



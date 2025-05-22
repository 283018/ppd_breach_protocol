# cpp/__init__.py
import os
import sys
import platform
import importlib.util


class BreachCppBuildError(ImportError):
    def __init__(self, original_error: Exception):
        original_message = str(original_error).strip()
        build_instruction = (
            "Build directory not found. Please build the module first:\n"
            r"'python .\cpp\setup.py build clean --all'"
            "    or    "
            r"'python3 ./cpp/setup.py build clean --all'"
        )
        full_message = f"{original_message}\n{build_instruction}"
        super().__init__(full_message)


system = platform.system()
plat_subdir = {"Windows": "win", "Darwin": "mac"}.get(system, "linux")
build_dir = os.path.join(os.path.dirname(__file__), "build", plat_subdir)

modules_to_load = {
    "ant_colony_cpp": "ant_colony",
    "breach_solver_cpp": "brute_force"
}

for module_name, func_name in modules_to_load.items():
    try:
        module_path = None
        for file in os.listdir(build_dir):
            if file.startswith(module_name) and (file.endswith(".pyd") or file.endswith(".so") or file.endswith(".dylib")):
                module_path = os.path.join(build_dir, file)
                break
        if not module_path:
            raise FileNotFoundError(f"No binary found for {module_name}")

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if not spec:
            raise ImportError(f"Failed to load module '{module_name}' from {module_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        if hasattr(module, func_name):
            globals()[func_name] = getattr(module, func_name)
        else:
            raise ImportError(f"Module '{module_name}' missing required function '{func_name}'")

    except Exception as e:
        raise BreachCppBuildError(e) from e

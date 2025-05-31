# install.py

import subprocess
import sys
import os

def run_command(command):
    print(f"Running: {' '.join(command)}")
    subprocess.check_call(command)

def main():
    run_command([sys.executable, '-m', 'pip', 'install', 'uv'])

    run_command([sys.executable, '-m', 'uv' 'pip', 'install', '-r', 'requirements.txt'])

    cpp_dir = os.path.join(os.getcwd(), 'cpp')
    if not os.path.exists(cpp_dir):
        raise FileNotFoundError(f"Directory {cpp_dir} not found.")

    os.chdir(cpp_dir)
    run_command([sys.executable, 'setup.py', 'build', 'clean', '--all'])

if __name__ == '__main__':
    main()
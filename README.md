# Cyberpunk 2077 Breach Protocol Solver (Educational Project)

This project is inspired by the **Breach Protocol minigame** from *Cyberpunk 2077*, a video game developed and published by **CD Projekt Red**. This repository is **not affiliated, endorsed, or sponsored** by CD Projekt Red or any related entities. All rights to *Cyberpunk 2077* and its intellectual property remain with the original creators.

---

## Gurobi Usage
This project uses the **Gurobi Optimizer** API for solving mathematical models. Gurobi is a proprietary solver, and its use here is strictly for **educational purposes** under the terms of the [Gurobi Academic License](https://www.gurobi.com/academia/academic-program-and-licenses/). 
- **Note**: Gurobi binaries or proprietary code are **not included** in this repository. Users must obtain their own Gurobi license (e.g., via an academic license) to run the solver.

## pybind11 Integration  
This project uses **[pybind11](https://github.com/pybind/pybind11 )** to interface between Python and C++ (performance-critical components). pybind11 is a header-only library that enables interoperability between Python and C++. pybind11 is used under the terms of the **BSD 3-Clause License**. For more information, see the [pybind11 license page](https://github.com/pybind/pybind11/blob/master/LICENSE ). 

- **Note**: pybind11 is **not included** in this repository and **not required to run** the project. It is only need to be installed separately if you want to **rebuild the C++ source code** into binary form.

---

## Project Purpose
This is a **non-commercial, educational project** created for a university assignment. It demonstrates the application of mathematical optimization (via Gurobi) to solve a puzzle inspired by the game.

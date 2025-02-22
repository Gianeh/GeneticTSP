# Parallel Genetic Algorithm for the Traveling Salesman Problem

## Project Overview
This project implements a parallel Genetic Algorithm (GA) to solve (suboptimally) the NP-hard Traveling Salesman Problem (TSP). The algorithm follows the Island Model paradigm and is parallelized for both multicore CPUs and GPUs using multithreading (with various techniques) and CUDA.

---

## Features
- **Genetic Algorithm (GA)**: Uses selection, crossover, mutation, and migration to evolve populations toward optimal solutions.
- **Island Model**: Isolates sub-populations (islands) to encourage diversity and enable parallelism.
- **Parallel Implementations**:
  - **CPU Multithreading**: Implements parallelism using raw threads, a thread pool, and OpenMP.
  - **GPU Acceleration**: Uses CUDA for efficient execution on NVIDIA GPUs.
- **Benchmarking**: Compares performance across different configurations with datasets from TSPLIB.

## Implementations
- **Single-Core CPU Version**: Sequential execution.
- **Multi-Core CPU Versions**:
  - Plain thread generation.
  - Thread pool-based parallelism.
  - OpenMP-based parallelism.
- **GPU Version (CUDA)**: Highly parallel execution at the island level.

## Results Summary
- **Performance Gains**:
  - OpenMP improves execution time significantly over the single-core approach.
  - CUDA accelerates execution further, especially for large populations.
- **Solution Quality**:
  - Larger populations lead to better solutions.
  - GPU implementation finds better solutions in a fixed time compared to CPU implementations.

---

## Installation and Usage
### Prerequisites
- **Compilers**:
  - GCC
  - NVCC (Tests were conducted with CUDA 12.3 for Ubuntu)
- **Libraries**:
  - OpenMP
  - pthreads (optional for CPU versions)
  - CUDA Toolkit (for GPU version)

### Running the Solver
To run the benchmarks, use the provided test scripts:
```sh
./run_tests.sh  # Runs standard benchmark tests
./further_tests.sh  # Runs extended tests focusing on CUDA and OpenMP
```
Each c, cpp or cuda file can be compiled with a variety of hyperparameter changes using the correct compiler| see #DEFINEs and //comments.

---

## Project Structure
```
├── data/              # Datasets (TSPLIB format)
├── results/           # Execution logs and plots
├── src/               # Source code
│   ├── Island_GA_single_core.c
│   ├── Island_GA_openmp.cpp
│   ├── Island_GA_cuda.cu
├── scripts/           # Python utilities for result visualization
├── README.md          # This file
└── run_tests.sh       # Script to execute benchmark tests
```

---

## References
[1] Prasanna Jog, Jung-Yul Suh, and Dirk Van Gucht. The effects of population size, heuristic crossover and local improvement on a genetic algorithm for the traveling salesman problem. 1989.

[2] Traveling salesman problem - cornell university computational optimization open textbook - optimization wiki.

[3] L. Wang, A.A. Maciejewski, H.J. Siegel, and V.P. Roychowdhury. A comparative study of five parallel genetic algorithms using the traveling salesman problem. In Proceedings of the First Merged International Parallel Processing Symposium and Symposium on Parallel and Distributed Processing, pages 345–349, 1998.

[4] John Runwei Cheng and Mitsuo Gen. Parallel genetic algorithms with gpu computing. In Tamás Bányai and Antonella Petrilloand Fabio De Felice, editors, Industry 4.0, chapter 6. IntechOpen, Rijeka, 2020.

[5] Boqun Wang, Hailong Zhang, Jun Nie, Jie Wang, Xinchen Ye, Toktonur Ergesh, Meng Zhang, Jia Li, and Wanqiong Wang. Multipopulation genetic algorithm based on gpu for solving tsp problem. Mathematical Problems in Engineering, 2020.

[6] Jean-Yves Potvin. Genetic algorithms for the traveling salesman problem. 2005.

*For more details, see the full project report.*

---


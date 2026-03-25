\# HPC Homework 5 — N-body (Serial + OpenMP)



This project benchmarks an N-body gravitational simulation:

1\) Serial C++ baseline (`src/nbody.cc`)

2\) OpenMP shared-memory version (`src/nbody\_omp.cc`)



\## Build (local or HPC)

Serial:

```bash

g++ -O3 -std=c++17 src/nbody.cc -o nbody

./nbody 128


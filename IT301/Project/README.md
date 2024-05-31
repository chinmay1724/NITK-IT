
# Hypercube Quicksort using MPI & Parallel Merge sort using OpenMP

This project
embarks on the exploration and implementation of parallel
sorting algorithms, focusing on Hypercube Quicksort utilizing
MPI (Message Passing Interface) and Merge Sort employing
OpenMP (Open Multi-Processing).


## How to Run the Project Codes

Follow Below Commands In Linux System for qsort_hypercube

```bash
  mpic++ qsort_hypercube.cpp -o qsort_hypercube    
  mpirun -np 2 qsort_hypercube 100 -1
```
Follow Below Commands In Linux System for sort_list_openmp

```bash
  gcc -fopenmp sort_list_openmp.c -o sort_list_openmp -lm    
  ./sort_list_openmp 10 2
```
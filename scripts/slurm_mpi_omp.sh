#!/bin/bash
#SBATCH -J hw5_hybrid
#SBATCH -A utdallas
#SBATCH -p cpu-preempt
#SBATCH -N 1
#SBATCH -n 64
#SBATCH -c 1
#SBATCH -t 00:20:00

set -euo pipefail
cd /scratch/ganymede2/tda230002/hw5/hpc-homework5-nbody

module purge
module load gnu12/12.4.0
module load openmpi4/4.1.6

# avoid UCX locked-memory issues
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader,tcp

# OpenMP binding (reduce oversubscription + keep stable timings)
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

mpicxx -O3 -std=c++17 -fopenmp src/nbody_mpi_omp.cc -o nbody_mpi_omp

mkdir -p results reports reports/data

N=1024
THREADS=4
OUT="reports/data/hybrid_times_N${N}_t${THREADS}.csv"
echo "ranks,threads,seconds" > "$OUT"

export OMP_NUM_THREADS=$THREADS

for p in 1 2 4 8 16; do
  echo "===== ranks=$p threads=$THREADS ====="
  mpirun -np $p -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES ./nbody_mpi_omp $N \
    | tee results/hybrid_run_p${p}_t${THREADS}.log

  sec=$(grep -oE 'Runtime = [0-9.]+ s' results/hybrid_run_p${p}_t${THREADS}.log | awk '{print $3}' | tail -n 1)
  echo "$p,$THREADS,$sec" >> "$OUT"
done

echo "Wrote $OUT"

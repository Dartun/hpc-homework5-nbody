#!/bin/bash
#SBATCH -J hw5_mpi_shared
#SBATCH -A utdallas
#SBATCH -p cpu-preempt
#SBATCH -N 1
#SBATCH -n 64
#SBATCH -c 1
#SBATCH -t 00:15:00

set -euo pipefail
cd /scratch/ganymede2/tda230002/hw5/hpc-homework5-nbody

module purge
module load gnu12/12.4.0
module load openmpi4/4.1.6

# Avoid UCX locked-memory issues on this cluster
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader,tcp

mkdir -p results reports reports/data

mpicxx -O3 -std=c++17 src/nbody_mpi_shared.cc -o nbody_mpi_shared

N=1024
OUT="reports/data/mpi_shared_times_N${N}.csv"
echo "ranks,seconds" > "$OUT"

for p in 1 2 4 8 16 32 64; do
  echo "===== shared ranks=$p ====="
  mpirun -np $p ./nbody_mpi_shared $N | tee results/mpi_shared_run_p${p}.log

  sec=$(grep -oE 'Runtime = [0-9.]+ s' results/mpi_shared_run_p${p}.log | awk '{print $3}' | tail -n 1)
  echo "$p,$sec" >> "$OUT"
done

echo "Wrote $OUT"

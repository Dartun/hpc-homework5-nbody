cat > scripts/bench_omp.sh << 'EOF'
#!/bin/bash
set -euo pipefail
mkdir -p results

g++ -O3 -std=c++17 -fopenmp src/nbody_omp.cc -o nbody_omp

N=1024
OUT="results/omp_times_N${N}.csv"
echo "threads,seconds" > "$OUT"

for th in 1 2 4 8 16 32 64; do
  export OMP_NUM_THREADS=$th
  echo "Running threads=$th"
  t=$(/usr/bin/time -f "%e" ./nbody_omp $N 2>&1 >/dev/null | tail -n 1)
  echo "$th,$t" >> "$OUT"
done

echo "Wrote $OUT"
EOF
chmod +x scripts/bench_omp.sh
cat > scripts/bench_serial.sh << 'EOF'
#!/bin/bash
set -euo pipefail
mkdir -p results

g++ -O3 -std=c++17 src/nbody.cc -o nbody

OUT="results/serial_times.csv"
echo "N,seconds" > "$OUT"

for N in 128 256 512 1024 2048; do
  echo "Running N=$N"
  t=$(/usr/bin/time -f "%e" ./nbody $N 2>&1 >/dev/null | tail -n 1)
  echo "$N,$t" >> "$OUT"
done

echo "Wrote $OUT"
EOF
chmod +x scripts/bench_serial.sh
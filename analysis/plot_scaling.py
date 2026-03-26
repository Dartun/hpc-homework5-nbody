import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("reports", exist_ok=True)

# ---------- Serial runtime vs N ----------
serial = pd.read_csv("results/serial_times.csv")
plt.figure()
plt.loglog(serial["N"], serial["seconds"], marker="o")
plt.xlabel("N")
plt.ylabel("Runtime (s)")
plt.title("Serial runtime vs N")
plt.tight_layout()
plt.savefig("reports/serial_runtime_vs_N.png", dpi=200)

# ---------- OpenMP timings ----------
omp = pd.read_csv("results/omp_times_N1024.csv")
t1_omp = float(omp.loc[omp["threads"] == 1, "seconds"].iloc[0])
omp["speedup"] = t1_omp / omp["seconds"]

# ---------- MPI timings ----------
mpi = pd.read_csv("reports/data/mpi_times_N1024.csv")
t1_mpi = float(mpi.loc[mpi["ranks"] == 1, "seconds"].iloc[0])
mpi["speedup"] = t1_mpi / mpi["seconds"]

# ---------- Comparison speedup plot ----------
plt.figure()
plt.plot(omp["threads"], omp["speedup"], marker="o", label="OpenMP speedup (N=1024)")
plt.plot(mpi["ranks"], mpi["speedup"], marker="o", label="MPI speedup (N=1024)")
plt.xlabel("Parallelism (threads or ranks)")
plt.ylabel("Speedup")
plt.title("Speedup comparison: OpenMP vs MPI")
plt.legend()
plt.tight_layout()
plt.savefig("reports/speedup_compare_omp_vs_mpi_N1024.png", dpi=200)

print("Wrote reports/speedup_compare_omp_vs_mpi_N1024.png")

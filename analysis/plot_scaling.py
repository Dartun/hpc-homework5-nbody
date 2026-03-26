import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("reports", exist_ok=True)

# ----- Serial log-log runtime vs N -----
serial = pd.read_csv("results/serial_times.csv")
plt.figure()
plt.loglog(serial["N"], serial["seconds"], marker="o")
plt.xlabel("N")
plt.ylabel("Runtime (s)")
plt.title("Serial runtime vs N")
plt.tight_layout()
plt.savefig("reports/serial_runtime_vs_N.png", dpi=200)

# ----- OpenMP runtime + speedup -----
omp = pd.read_csv("results/omp_times_N1024.csv")
t1 = float(omp.loc[omp["threads"] == 1, "seconds"].iloc[0])
omp["speedup"] = t1 / omp["seconds"]
omp["efficiency"] = omp["speedup"] / omp["threads"]

plt.figure()
plt.plot(omp["threads"], omp["seconds"], marker="o")
plt.xlabel("Threads")
plt.ylabel("Runtime (s)")
plt.title("OpenMP runtime vs threads (N=1024)")
plt.tight_layout()
plt.savefig("reports/omp_runtime_N1024.png", dpi=200)

plt.figure()
plt.plot(omp["threads"], omp["speedup"], marker="o")
plt.xlabel("Threads")
plt.ylabel("Speedup")
plt.title("OpenMP speedup vs threads (N=1024)")
plt.tight_layout()
plt.savefig("reports/omp_speedup_N1024.png", dpi=200)

plt.figure()
plt.plot(omp["threads"], omp["efficiency"], marker="o")
plt.xlabel("Threads")
plt.ylabel("Efficiency")
plt.title("OpenMP efficiency vs threads (N=1024)")
plt.tight_layout()
plt.savefig("reports/omp_efficiency_N1024.png", dpi=200)

print("Wrote plots to reports/:")
for fn in [
    "serial_runtime_vs_N.png",
    "omp_runtime_N1024.png",
    "omp_speedup_N1024.png",
    "omp_efficiency_N1024.png",
]:
    print("  -", os.path.join("reports", fn))

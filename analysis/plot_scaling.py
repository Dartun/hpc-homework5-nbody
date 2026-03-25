cat > analysis/plot_scaling.py << 'EOF'
import pandas as pd
import matplotlib.pyplot as plt

def plot_serial():
    df = pd.read_csv("results/serial_times.csv")
    plt.figure()
    plt.loglog(df["N"], df["seconds"], marker="o")
    plt.xlabel("N")
    plt.ylabel("Runtime (s)")
    plt.title("Serial runtime vs N")
    plt.tight_layout()
    plt.savefig("reports/serial_runtime_vs_N.png", dpi=200)

def plot_omp(path):
    df = pd.read_csv(path)
    t1 = df.loc[df["threads"]==1, "seconds"].iloc[0]
    df["speedup"] = t1 / df["seconds"]
    plt.figure()
    plt.plot(df["threads"], df["seconds"], marker="o")
    plt.xlabel("Threads")
    plt.ylabel("Runtime (s)")
    plt.title("OpenMP runtime vs threads")
    plt.tight_layout()
    plt.savefig("reports/omp_runtime.png", dpi=200)

    plt.figure()
    plt.plot(df["threads"], df["speedup"], marker="o")
    plt.xlabel("Threads")
    plt.ylabel("Speedup")
    plt.title("OpenMP speedup vs threads")
    plt.tight_layout()
    plt.savefig("reports/omp_speedup.png", dpi=200)

if __name__ == "__main__":
    import os
    os.makedirs("reports", exist_ok=True)
    plot_serial()
    # update filename if you change N
    plot_omp("results/omp_times_N1024.csv")
    print("Wrote plots to reports/")
EOF
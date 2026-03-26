# analysis/plot_scaling.py
import os
import pandas as pd
import matplotlib.pyplot as plt


def load_times_csv(path, col_ranks="ranks", col_seconds="seconds"):
    df = pd.read_csv(path)
    # basic sanity
    if col_ranks not in df.columns or col_seconds not in df.columns:
        raise ValueError(f"{path} must contain columns: {col_ranks}, {col_seconds}")
    df = df.sort_values(col_ranks).reset_index(drop=True)
    return df


def add_speedup(df, col_ranks="ranks", col_seconds="seconds", out_col="speedup"):
    t1 = float(df.loc[df[col_ranks] == 1, col_seconds].iloc[0])
    df[out_col] = t1 / df[col_seconds]
    return df


def main():
    os.makedirs("reports", exist_ok=True)

    # -------- Load OpenMP --------
    omp_path = "results/omp_times_N1024.csv"
    if os.path.exists(omp_path):
        omp = load_times_csv(omp_path, col_ranks="threads", col_seconds="seconds")
        omp = add_speedup(omp, col_ranks="threads", col_seconds="seconds", out_col="speedup")
    else:
        omp = None
        print(f"WARNING: missing {omp_path}")

    # -------- Load MPI --------
    mpi_path = "reports/data/mpi_times_N1024.csv"
    if os.path.exists(mpi_path):
        mpi = load_times_csv(mpi_path, col_ranks="ranks", col_seconds="seconds")
        mpi = add_speedup(mpi, col_ranks="ranks", col_seconds="seconds", out_col="speedup")
    else:
        mpi = None
        print(f"WARNING: missing {mpi_path}")

    # -------- Load MPI Shared (Section 5) --------
    mpi_shared_path = "reports/data/mpi_shared_times_N1024.csv"
    if os.path.exists(mpi_shared_path):
        mpi_shared = load_times_csv(mpi_shared_path, col_ranks="ranks", col_seconds="seconds")
        mpi_shared = add_speedup(mpi_shared, col_ranks="ranks", col_seconds="seconds", out_col="speedup")
    else:
        mpi_shared = None
        print(f"WARNING: missing {mpi_shared_path}")

    # -------- Plot: speedup comparison --------
    plt.figure()
    if omp is not None:
        plt.plot(omp["threads"], omp["speedup"], marker="o", label="OpenMP speedup (N=1024)")
    if mpi is not None:
        plt.plot(mpi["ranks"], mpi["speedup"], marker="o", label="MPI speedup (N=1024)")
    if mpi_shared is not None:
        plt.plot(mpi_shared["ranks"], mpi_shared["speedup"], marker="o", label="MPI shared speedup (N=1024)")

    plt.xlabel("Parallelism (threads or ranks)")
    plt.ylabel("Speedup (T1 / Tp)")
    plt.title("Speedup comparison: OpenMP vs MPI vs MPI Shared")
    plt.legend()
    plt.tight_layout()

    out_png = "reports/speedup_compare_all_N1024.png"
    plt.savefig(out_png, dpi=200)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()

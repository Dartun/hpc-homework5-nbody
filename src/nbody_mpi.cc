#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <fstream>
#include <string>
#include <cstdlib>

using std::vector;

// ---------- global problem params (match serial) ----------
static int    N  = 128;
static int    D  = 3;
static int    ND = N * D;

static double G  = 0.5;
static double dt = 1e-3;
static int    T  = 300;

static double x_min = 0.0, x_max = 1.0;
static double v_min = 0.0, v_max = 0.0;

static double m0 = 1.0;
static double epsilon = 0.01;
static double epsilon2 = epsilon * epsilon;

// ---------- MPI globals ----------
static int rank = 0, n_ranks = 1;

static vector<int> counts, displs;     // masses per rank
static vector<int> countsD, displsD;   // state counts ( * D ) per rank

static int N_beg=0, N_end=0, N_local=0;
static int ND_beg=0, ND_end=0, ND_local=0;

static std::mt19937 gen;

// ---------- IO helpers ----------
static void save_vec(const vector<double>& vec, const std::string& filename, const std::string& header="") {
    std::ofstream f(filename);
    if (!f.is_open()) {
        if (rank == 0) std::cerr << "Unable to open " << filename << "\n";
        return;
    }
    if (!header.empty()) f << header << "\n";
    for (double x : vec) f << x << "\n";
}

static void setup_parallelism() {
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    // deterministic but different per-rank seed
    auto now = std::chrono::high_resolution_clock::now();
    auto now_us = std::chrono::time_point_cast<std::chrono::microseconds>(now);
    long long now_int = now_us.time_since_epoch().count();
    gen.seed(static_cast<unsigned int>(now_int ^ (long long)rank));

    counts.assign(n_ranks, 0);
    displs.assign(n_ranks, 0);
    countsD.assign(n_ranks, 0);
    displsD.assign(n_ranks, 0);

    int base = N / n_ranks;
    int rem  = N % n_ranks;

    for (int r = 0; r < n_ranks; ++r) {
        counts[r] = base + (r < rem ? 1 : 0);
        displs[r] = (r == 0) ? 0 : (displs[r-1] + counts[r-1]);
        countsD[r] = counts[r] * D;
        displsD[r] = displs[r] * D;
    }

    N_beg = displs[rank];
    N_end = N_beg + counts[rank];
    N_local = counts[rank];

    ND_beg = N_beg * D;
    ND_end = N_end * D;
    ND_local = N_local * D;
}

static void initial_conditions(vector<double>& x, vector<double>& v) {
    std::uniform_real_distribution<double> ran01(0.0, 1.0);
    const double dx = x_max - x_min;
    const double dv = v_max - v_min;

    // Only initialize local slice, then allgather to replicate full arrays on each rank.
    for (int i = ND_beg; i < ND_end; ++i) {
        x[i] = x_min + dx * ran01(gen);
        v[i] = v_min + dv * ran01(gen);
    }

    if (n_ranks > 1) {
        MPI_Allgatherv(x.data() + ND_beg, ND_local, MPI_DOUBLE,
                       x.data(), countsD.data(), displsD.data(), MPI_DOUBLE, MPI_COMM_WORLD);

        MPI_Allgatherv(v.data() + ND_beg, ND_local, MPI_DOUBLE,
                       v.data(), countsD.data(), displsD.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    }
}

static void compute_accel(const vector<double>& x, const vector<double>& m, vector<double>& a_local) {
    // compute accel only for local masses [N_beg, N_end)
    std::fill(a_local.begin(), a_local.end(), 0.0);

    for (int i = N_beg; i < N_end; ++i) {
        int iD = i * D;
        double ax0=0.0, ax1=0.0, ax2=0.0;

        for (int j = 0; j < N; ++j) {
            int jD = j * D;
            double dx0 = x[jD+0] - x[iD+0];
            double dx1 = x[jD+1] - x[iD+1];
            double dx2 = x[jD+2] - x[iD+2];
            double r2  = dx0*dx0 + dx1*dx1 + dx2*dx2 + epsilon2;
            double invr3 = 1.0 / (r2 * std::sqrt(r2));
            double c = G * m[j] * invr3;
            ax0 += c * dx0;
            ax1 += c * dx1;
            ax2 += c * dx2;
        }

        int li = (i - N_beg) * D; // local index in a_local
        a_local[li+0] = ax0;
        a_local[li+1] = ax1;
        a_local[li+2] = ax2;
    }
}

static void timestep(vector<double>& x, vector<double>& v, const vector<double>& m) {
    vector<double> a_local(ND_local, 0.0);
    compute_accel(x, m, a_local);

    // update only local slice into temp arrays
    vector<double> x1_local(ND_local), v1_local(ND_local);

    for (int i = N_beg; i < N_end; ++i) {
        int iD = i * D;
        int li = (i - N_beg) * D;

        v1_local[li+0] = v[iD+0] + dt * a_local[li+0];
        v1_local[li+1] = v[iD+1] + dt * a_local[li+1];
        v1_local[li+2] = v[iD+2] + dt * a_local[li+2];

        x1_local[li+0] = x[iD+0] + dt * v1_local[li+0];
        x1_local[li+1] = x[iD+1] + dt * v1_local[li+1];
        x1_local[li+2] = x[iD+2] + dt * v1_local[li+2];
    }

    // gather updated x,v to all ranks (so next accel sees full arrays)
    if (n_ranks > 1) {
        MPI_Allgatherv(x1_local.data(), ND_local, MPI_DOUBLE,
                       x.data(), countsD.data(), displsD.data(), MPI_DOUBLE, MPI_COMM_WORLD);

        MPI_Allgatherv(v1_local.data(), ND_local, MPI_DOUBLE,
                       v.data(), countsD.data(), displsD.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    } else {
        // rank=0 only: just copy local into full arrays
        for (int i = N_beg; i < N_end; ++i) {
            int iD = i * D;
            int li = (i - N_beg) * D;
            x[iD+0] = x1_local[li+0];
            x[iD+1] = x1_local[li+1];
            x[iD+2] = x1_local[li+2];
            v[iD+0] = v1_local[li+0];
            v[iD+1] = v1_local[li+1];
            v[iD+2] = v1_local[li+2];
        }
    }
}

static double kinetic_energy_local(const vector<double>& v, const vector<double>& m) {
    double KE = 0.0;
    for (int i = N_beg; i < N_end; ++i) {
        int iD = i * D;
        double v2 = v[iD+0]*v[iD+0] + v[iD+1]*v[iD+1] + v[iD+2]*v[iD+2];
        KE += 0.5 * m[i] * v2;
    }
    return KE;
}

int main(int argc, char** argv) {
    if (argc > 1) {
        N = std::atoi(argv[1]);
        ND = N * D;
    }

    setup_parallelism();

    vector<double> x(ND, 0.0), v(ND, 0.0);
    vector<double> m(N, m0);

    initial_conditions(x, v);

    if (rank == 0) std::system("mkdir -p results reports");

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    vector<double> KE_root;
    if (rank == 0) KE_root.assign(T+1, 0.0);

    for (int t = 0; t <= T; ++t) {
        double ke_loc = kinetic_energy_local(v, m);
        double ke_sum = 0.0;
        MPI_Reduce(&ke_loc, &ke_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) KE_root[t] = ke_sum;

        if (t < T) timestep(x, v, m);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    double elapsed = t1 - t0;

    // report max rank time (more honest)
    double elapsed_max = 0.0;
    MPI_Reduce(&elapsed, &elapsed_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Runtime = " << elapsed_max << " s for N = " << N
                  << " with MPI ranks=" << n_ranks << "\n";
        save_vec(KE_root, "results/ke_mpi_N" + std::to_string(N) + ".txt", "KineticEnergy");
    }

    MPI_Finalize();
    return 0;
}

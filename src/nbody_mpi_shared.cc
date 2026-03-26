// src/nbody_mpi_shared.cc
//
// MPI Shared-Memory N-body (single-node).
// Uses MPI_Win_allocate_shared so x, v, a are shared among ranks on the node.
// Each rank updates only its chunk; barriers synchronize steps.

#include <mpi.h>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// -------------------- Problem constants (match your serial/OpenMP code) --------------------
static int    N  = 128;
static const int D  = 3;
static int    ND = N * D;

static const double G       = 0.5;
static const double dt      = 1e-3;
static const int    T       = 300;

static const double x_min   = 0.0;
static const double x_max   = 1.0;
static const double v_min   = 0.0;
static const double v_max   = 0.0;

// Softening to avoid r->0 blowup
static const double epsilon  = 0.01;
static const double eps2     = epsilon * epsilon;

// -------------------- MPI globals --------------------
static int rank = 0;
static int n_ranks = 1;

// Split N across ranks
static int N_beg = 0, N_end = 0, N_local = 0;
static int ND_beg = 0, ND_end = 0, ND_local = 0;

// Shared-memory communicator (all ranks on the same node)
static MPI_Comm shmcomm = MPI_COMM_NULL;

// Shared-memory windows + pointers
static MPI_Win win_m      = MPI_WIN_NULL;
static MPI_Win win_x      = MPI_WIN_NULL;
static MPI_Win win_v      = MPI_WIN_NULL;
static MPI_Win win_a      = MPI_WIN_NULL;
static MPI_Win win_x_next = MPI_WIN_NULL;
static MPI_Win win_v_next = MPI_WIN_NULL;

static double *m      = nullptr;
static double *x      = nullptr;
static double *v      = nullptr;
static double *a      = nullptr;
static double *x_next = nullptr;
static double *v_next = nullptr;

// -------------------- Helpers --------------------
static void save_vec(const std::vector<double>& vec, const std::string& filename, const std::string& header="") {
    std::ofstream f(filename);
    if (!f) {
        std::cerr << "Unable to open file " << filename << "\n";
        return;
    }
    if (!header.empty()) f << header << "\n";
    for (double val : vec) f << val << "\n";
}

static inline double now_seconds() {
    using clk = std::chrono::high_resolution_clock;
    return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}

// Allocate one shared array. Rank 0 allocates real bytes; others allocate 0 and query.
static double* alloc_shared_array(MPI_Win &win, MPI_Aint n_doubles) {
    MPI_Aint bytes = 0;
    int disp_unit = sizeof(double);

    if (rank == 0) bytes = n_doubles * (MPI_Aint)sizeof(double);

    double* baseptr = nullptr;
    MPI_Win_allocate_shared(bytes, disp_unit, MPI_INFO_NULL, shmcomm, &baseptr, &win);

    if (rank != 0) {
        // Query rank 0’s segment
        MPI_Aint sz = 0;
        int du = 0;
        double* p = nullptr;
        MPI_Win_shared_query(win, 0, &sz, &du, &p);
        baseptr = p;
    }

    // Ensure everyone sees it
    MPI_Barrier(shmcomm);

    // Zero-initialize once (rank 0)
    if (rank == 0 && bytes > 0) {
        std::memset(baseptr, 0, (size_t)bytes);
    }
    MPI_Barrier(shmcomm);

    return baseptr;
}

static void setup_parallelism() {
    MPI_Init(nullptr, nullptr);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    // All ranks on one node for shared memory windows
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm);

    // Split N across ranks (block distribution with remainder)
    const int q = N / n_ranks;
    const int r = N % n_ranks;

    N_local = q + (rank < r ? 1 : 0);
    N_beg   = rank * q + (rank < r ? rank : r);
    N_end   = N_beg + N_local;

    ND_beg   = N_beg * D;
    ND_end   = N_end * D;
    ND_local = ND_end - ND_beg;
}

static void allocate_shared_state() {
    // allocate shared arrays
    m      = alloc_shared_array(win_m,      (MPI_Aint)N);
    x      = alloc_shared_array(win_x,      (MPI_Aint)ND);
    v      = alloc_shared_array(win_v,      (MPI_Aint)ND);
    a      = alloc_shared_array(win_a,      (MPI_Aint)ND);
    x_next = alloc_shared_array(win_x_next, (MPI_Aint)ND);
    v_next = alloc_shared_array(win_v_next, (MPI_Aint)ND);
}

static void free_shared_state() {
    // Free in reverse-ish order, barriers to be safe
    MPI_Barrier(shmcomm);
    if (win_v_next != MPI_WIN_NULL) MPI_Win_free(&win_v_next);
    if (win_x_next != MPI_WIN_NULL) MPI_Win_free(&win_x_next);
    if (win_a      != MPI_WIN_NULL) MPI_Win_free(&win_a);
    if (win_v      != MPI_WIN_NULL) MPI_Win_free(&win_v);
    if (win_x      != MPI_WIN_NULL) MPI_Win_free(&win_x);
    if (win_m      != MPI_WIN_NULL) MPI_Win_free(&win_m);

    if (shmcomm != MPI_COMM_NULL) MPI_Comm_free(&shmcomm);
    MPI_Finalize();
}

// Initialize only local chunk, then barrier so all ranks see full arrays.
static void initial_conditions_shared() {
    // Seed RNG differently per rank
    const auto seed = (unsigned long long)(now_seconds() * 1e6) ^ (unsigned long long)(rank * 0x9e3779b97f4a7c15ULL);
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<double> rx(x_min, x_max);
    std::uniform_real_distribution<double> rv(v_min, v_max);

    // Masses (all = 1) — rank initializes its local part
    for (int i = N_beg; i < N_end; ++i) m[i] = 1.0;

    // Positions and velocities for local particles
    for (int i = N_beg; i < N_end; ++i) {
        const int base = i * D;
        for (int k = 0; k < D; ++k) {
            x[base + k] = rx(gen);
            v[base + k] = rv(gen);
        }
    }

    MPI_Barrier(shmcomm);
}

// Compute acceleration for local i range using shared x,m (full arrays available).
static void compute_accel_shared() {
    // zero local accel
    for (int idx = ND_beg; idx < ND_end; ++idx) a[idx] = 0.0;

    for (int i = N_beg; i < N_end; ++i) {
        const int iD = i * D;
        for (int j = 0; j < N; ++j) {
            const int jD = j * D;

            double dx0 = x[jD + 0] - x[iD + 0];
            double dx1 = x[jD + 1] - x[iD + 1];
            double dx2 = x[jD + 2] - x[iD + 2];

            double r2 = dx0*dx0 + dx1*dx1 + dx2*dx2 + eps2;
            double invr = 1.0 / std::sqrt(r2);
            double invr3 = invr * invr * invr;

            double coef = G * m[j] * invr3; // G*mj/|r|^3
            a[iD + 0] += coef * dx0;
            a[iD + 1] += coef * dx1;
            a[iD + 2] += coef * dx2;
        }
    }
}

// One timestep: compute accel, then update local chunk into x_next/v_next.
// Then barrier + swap pointers (all ranks do the same swap).
static void timestep_shared() {
    compute_accel_shared();

    for (int i = N_beg; i < N_end; ++i) {
        const int base = i * D;
        for (int k = 0; k < D; ++k) {
            v_next[base + k] = v[base + k] + dt * a[base + k];
            x_next[base + k] = x[base + k] + dt * v_next[base + k];
        }
    }

    MPI_Barrier(shmcomm);

    // swap (all ranks)
    std::swap(x, x_next);
    std::swap(v, v_next);

    MPI_Barrier(shmcomm);
}

// Kinetic energy for local chunk (sum_i 0.5*m_i*|v_i|^2)
static double kinetic_energy_local() {
    double ke = 0.0;
    for (int i = N_beg; i < N_end; ++i) {
        const int base = i * D;
        double v2 = 0.0;
        for (int k = 0; k < D; ++k) v2 += v[base + k] * v[base + k];
        ke += 0.5 * m[i] * v2;
    }
    return ke;
}

int main(int argc, char** argv) {
    if (argc > 1) {
        N = std::atoi(argv[1]);
        ND = N * D;
    }

    setup_parallelism();
    allocate_shared_state();

    if (rank == 0) {
        std::cout << "MPI ranks = " << n_ranks << " (shared memory)\n";
        std::cout << "N = " << N << ", D = " << D << ", T = " << T << ", dt = " << dt << "\n";
    }

    initial_conditions_shared();

    // time vector + KE recorded on root only
    std::vector<double> tvec;
    std::vector<double> KE;
    if (rank == 0) {
        tvec.resize(T + 1);
        KE.resize(T + 1);
        for (int n = 0; n <= T; ++n) tvec[n] = n * dt;
    }

    MPI_Barrier(shmcomm);
    const auto start = std::chrono::high_resolution_clock::now();

    // step loop
    for (int n = 0; n <= T; ++n) {
        // KE at current step
        double ke_loc = kinetic_energy_local();
        double ke_sum = 0.0;
        MPI_Reduce(&ke_loc, &ke_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) KE[n] = ke_sum;

        if (n < T) {
            timestep_shared();
        }
    }

    MPI_Barrier(shmcomm);
    const auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    if (rank == 0) {
        // Save
        const std::string ke_file = "results/ke_mpi_shared_N" + std::to_string(N) + ".txt";
        const std::string t_file  = "results/time_shared_N" + std::to_string(N) + ".txt";
        save_vec(KE,   ke_file, "KineticEnergy");
        save_vec(tvec, t_file,  "Time");

        std::cout << "Runtime = " << elapsed << " s for N = " << N
                  << " with MPI ranks=" << n_ranks << " (shared)\n";
    }

    free_shared_state();
    return 0;
}

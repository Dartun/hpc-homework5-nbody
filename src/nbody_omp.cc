#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <fstream>
#include <string>

#include <omp.h>

using std::vector;

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

// Thread-private RNG
static std::mt19937 gen;
static std::uniform_real_distribution<double> ran01(0.0, 1.0);
#pragma omp threadprivate(gen)

static void save_vec(const vector<double>& vec, const std::string& filename, const std::string& header="") {
    std::ofstream f(filename);
    if (!f.is_open()) {
        std::cerr << "Unable to open " << filename << "\n";
        return;
    }
    if (!header.empty()) f << header << "\n";
    for (double x : vec) f << x << "\n";
}

static void initial_conditions(vector<double>& x, vector<double>& v) {
    const double dx = x_max - x_min;
    const double dv = v_max - v_min;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        gen.seed(123 + tid);

        #pragma omp for
        for (int i = 0; i < ND; ++i) {
            x[i] = x_min + dx * ran01(gen);
            v[i] = v_min + dv * ran01(gen);
        }
    }
}

static void compute_accel(const vector<double>& x, const vector<double>& m, vector<double>& a) {
    std::fill(a.begin(), a.end(), 0.0);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
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
        a[iD+0] = ax0;
        a[iD+1] = ax1;
        a[iD+2] = ax2;
    }
}

static void timestep(vector<double>& x, vector<double>& v, const vector<double>& m) {
    vector<double> a(ND, 0.0);
    compute_accel(x, m, a);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < ND; ++i) {
        v[i] += dt * a[i];
        x[i] += dt * v[i];
    }
}

static double kinetic_energy(const vector<double>& v, const vector<double>& m) {
    double KE = 0.0;

    #pragma omp parallel for reduction(+:KE) schedule(static)
    for (int i = 0; i < N; ++i) {
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

    vector<double> x(ND), v(ND);
    vector<double> m(N, m0);

    initial_conditions(x, v);

    auto start = std::chrono::high_resolution_clock::now();

    vector<double> KE(T+1);
    for (int t = 0; t <= T; ++t) {
        KE[t] = kinetic_energy(v, m);
        if (t < T) timestep(x, v, m);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    std::cout << "Runtime = " << elapsed << " s for N = " << N
              << " with OMP_NUM_THREADS=" << omp_get_max_threads() << "\n";

    save_vec(KE, "results/ke_omp_N" + std::to_string(N) + ".txt", "KineticEnergy");
    return 0;
}
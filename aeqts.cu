// aeqts.cu — AEQTS CUDA solver for Knapsack QUBO (single-run, JSON to stdout)
//
// Build:
//   nvcc -O3 -std=c++17 aeqts.cu -o solve_cuda
//
// Usage:
//   ./solve_cuda \
//     --weights  "1.0,2.0,3.0" \
//     --values   "4.0,5.0,6.0" \
//     --capacity 5.0 \
//     --penalty  10  \
//     --population 50 \
//     --iterations 1000 \
//     [--seed 42]
//
// Output: single JSON object to stdout; progress/errors to stderr.

#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <string>
#include <exception>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

using namespace std;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ── CUDA error check ──────────────────────────────────────────────────────────
#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        std::cerr << "[CUDA ERROR] " << __FILE__ << ":" << __LINE__ \
                  << " : " << cudaGetErrorString(_e) << std::endl; \
        std::exit(1); \
    } \
} while(0)

// ── Host RNG ─────────────────────────────────────────────────────────────────
static mt19937 g_rng;

// ── CUDA kernels (identical to test.cu) ──────────────────────────────────────

__global__ void init_rng_kernel(curandStatePhilox4_32_10_t* st,
                                unsigned long long seed, int total) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid < total)
        curand_init(seed, (unsigned long long)tid, 0ULL, &st[tid]);
}

__global__ void init_indices_kernel(int* idx, int N) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid < N) idx[tid] = tid;
}

__global__ void gen_neighbours_kernel(
    const double* __restrict__ q, int n_items,
    unsigned char* __restrict__ neighbours, int N,
    curandStatePhilox4_32_10_t* __restrict__ rng)
{
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= N * n_items) return;
    int i = tid % n_items;
    double beta = q[2 * i + 1];
    double prob = beta * beta;
    float  u    = curand_uniform(&rng[tid]);
    neighbours[tid] = (u <= (float)prob) ? 1 : 0;
}

__global__ void energy_kernel(
    const unsigned char* __restrict__ neighbours, int N, int n_items,
    const double* __restrict__ Q,
    double* __restrict__ energies)
{
    int n = (int)blockIdx.x;
    if (n >= N) return;
    const unsigned char* x = neighbours + n * n_items;
    double local = 0.0;
    for (int i = (int)threadIdx.x; i < n_items; i += (int)blockDim.x) {
        if (!x[i]) continue;
        const double* Qi = Q + i * n_items;
        double rowSum = 0.0;
        for (int j = 0; j < n_items; ++j)
            if (x[j]) rowSum += Qi[j];
        local += rowSum;
    }
    extern __shared__ double sh[];
    int lane = (int)threadIdx.x;
    sh[lane] = local;
    __syncthreads();
    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) sh[lane] += sh[lane + stride];
        __syncthreads();
    }
    if (lane == 0) energies[n] = sh[0];
}

__global__ void updateQ_pairs_kernel(
    const unsigned char* __restrict__ neighbours, int n_items, int N,
    const int* __restrict__ sorted_idx,
    const double* __restrict__ base_theta_table,
    double* __restrict__ q)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n_items) return;
    int num_pairs = N / 2;
    double raw = 0.0;
    for (int r = 0; r < num_pairs; ++r) {
        int bi = sorted_idx[r];
        int wi = sorted_idx[N - 1 - r];
        int diff = (int)neighbours[bi * n_items + i] - (int)neighbours[wi * n_items + i];
        raw += (double)diff * base_theta_table[r];
    }
    double alpha      = q[2 * i + 0];
    double beta       = q[2 * i + 1];
    double sign_mask  = (alpha * beta < 0.0) ? -1.0 : 1.0;
    double final_theta = raw * sign_mask;
    double s, c;
    sincos(final_theta, &s, &c);
    q[2 * i + 0] = alpha * c - beta * s;
    q[2 * i + 1] = alpha * s + beta * c;
}

// ── GPU context ───────────────────────────────────────────────────────────────
struct GPUContext {
    int n_items = 0, N = 0;
    double*        d_Q          = nullptr;
    double*        d_q          = nullptr;
    unsigned char* d_neigh      = nullptr;
    double*        d_energy     = nullptr;
    int*           d_idx        = nullptr;
    double*        d_base_theta = nullptr;
    curandStatePhilox4_32_10_t* d_rng = nullptr;

    void free_all() {
        if (d_Q)          CUDA_CHECK(cudaFree(d_Q));
        if (d_q)          CUDA_CHECK(cudaFree(d_q));
        if (d_neigh)      CUDA_CHECK(cudaFree(d_neigh));
        if (d_energy)     CUDA_CHECK(cudaFree(d_energy));
        if (d_idx)        CUDA_CHECK(cudaFree(d_idx));
        if (d_base_theta) CUDA_CHECK(cudaFree(d_base_theta));
        if (d_rng)        CUDA_CHECK(cudaFree(d_rng));
        d_Q = d_q = d_energy = d_base_theta = nullptr;
        d_neigh = nullptr; d_idx = nullptr; d_rng = nullptr;
        n_items = N = 0;
    }
    ~GPUContext() { free_all(); }

    void init(int n_items_, int N_) {
        free_all();
        n_items = n_items_; N = N_;
        CUDA_CHECK(cudaMalloc(&d_Q,          (size_t)n_items * n_items * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_q,          (size_t)2 * n_items       * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_neigh,      (size_t)N * n_items        * sizeof(unsigned char)));
        CUDA_CHECK(cudaMalloc(&d_energy,     (size_t)N                  * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_idx,        (size_t)N                  * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_base_theta, (size_t)(N / 2)            * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_rng,        (size_t)N * n_items * sizeof(curandStatePhilox4_32_10_t)));
    }
};

// ── QUBO builder (knapsack, same formula as test.cu) ─────────────────────────
vector<double> build_qubo_flat(const vector<double>& weights,
                               const vector<double>& values,
                               double capacity, double penalty) {
    int n = (int)values.size();
    vector<double> Q(n * n, 0.0);
    for (int i = 0; i < n; ++i) {
        double w = weights[i];
        // diagonal: -value + P*(w^2 - 2*cap*w)
        Q[i * n + i] = -values[i] + penalty * (w * w - 2.0 * capacity * w);
        for (int j = i + 1; j < n; ++j) {
            double qij = penalty * weights[i] * weights[j]; // 2*P*wi*wj / 2 → P*wi*wj (symmetric split)
            Q[i * n + j] = qij;
            Q[j * n + i] = qij;
        }
    }
    return Q;
}

// ── Shannon entropy from flat Q-bit state ────────────────────────────────────
double calc_entropy(const vector<double>& qflat) {
    double total = 0.0;
    int n = (int)qflat.size() / 2;
    for (int i = 0; i < n; ++i) {
        double a = qflat[2 * i], b = qflat[2 * i + 1];
        double p0 = a * a, p1 = b * b;
        if (p0 > 1e-15 && p1 > 1e-15)
            total += -(p0 * log2(p0) + p1 * log2(p1));
    }
    return total / n;
}

// ── CSV parser ────────────────────────────────────────────────────────────────
vector<double> parse_csv(const string& s) {
    vector<double> v;
    istringstream ss(s);
    string tok;
    while (getline(ss, tok, ',')) {
        if (!tok.empty()) v.push_back(stod(tok));
    }
    return v;
}

// ── History record ────────────────────────────────────────────────────────────
struct HistoryPoint {
    int    iteration;
    double energy;          // running best QUBO energy
    double current_energy;  // this iteration's best neighbour energy
    double objective;       // total value of global best solution
    double entropy;
    bool   is_feasible;
};

// ── Helpers for JSON output ───────────────────────────────────────────────────
static string jbool(bool b) { return b ? "true" : "false"; }
static string jdbl(double d) {
    ostringstream os;
    os << setprecision(10) << d;
    return os.str();
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    // ── Parse CLI args ────────────────────────────────────────────────────────
    vector<double> weights, values;
    double capacity   = 0.0;
    double penalty    = 10.0;
    int    population = 50;
    int    iterations = 1000;
    long long seed    = -1LL;

    for (int i = 1; i < argc - 1; ++i) {
        string key = argv[i];
        string val = argv[i + 1];
        if      (key == "--weights")    { weights    = parse_csv(val); ++i; }
        else if (key == "--values")     { values     = parse_csv(val); ++i; }
        else if (key == "--capacity")   { capacity   = stod(val);       ++i; }
        else if (key == "--penalty")    { penalty    = stod(val);       ++i; }
        else if (key == "--population") { population = stoi(val);       ++i; }
        else if (key == "--iterations") { iterations = stoi(val);       ++i; }
        else if (key == "--seed")       { seed       = stoll(val);      ++i; }
    }

    if (weights.empty() || values.empty() || weights.size() != values.size()) {
        cerr << "[aeqts] Error: --weights and --values must be non-empty and equal length\n";
        return 1;
    }
    if (population < 2 || population % 2 != 0) {
        cerr << "[aeqts] Error: --population must be an even integer >= 2\n";
        return 1;
    }

    int n = (int)weights.size();
    cerr << "[aeqts] n_items=" << n << " population=" << population
         << " iterations=" << iterations << " penalty=" << penalty
         << " capacity=" << capacity << "\n";

    // ── Seed ─────────────────────────────────────────────────────────────────
    if (seed < 0LL) {
        seed = (long long)chrono::high_resolution_clock::now().time_since_epoch().count();
    }
    g_rng.seed((unsigned long long)seed);

    // ── Build QUBO matrix ────────────────────────────────────────────────────
    vector<double> Qflat = build_qubo_flat(weights, values, capacity, penalty);

    // ── Init GPU ─────────────────────────────────────────────────────────────
    GPUContext gpu;
    gpu.init(n, population);
    CUDA_CHECK(cudaMemcpy(gpu.d_Q, Qflat.data(),
                          (size_t)n * n * sizeof(double), cudaMemcpyHostToDevice));

    // ── Pick random theta (same as original test.cu) ──────────────────────────
    vector<double> theta_list;
    for (int i = 1; i <= 10; ++i) theta_list.push_back(i * 0.01);
    uniform_int_distribution<> tdist(0, (int)theta_list.size() - 1);
    double current_theta = theta_list[tdist(g_rng)];
    cerr << "[aeqts] theta=" << current_theta << "\n";

    // ── Init Q-bits: α = β = 1/√2 ────────────────────────────────────────────
    double sq2 = 1.0 / sqrt(2.0);
    vector<double> qflat(2 * n, sq2);
    CUDA_CHECK(cudaMemcpy(gpu.d_q, qflat.data(),
                          qflat.size() * sizeof(double), cudaMemcpyHostToDevice));

    // ── Init GPU RNG ─────────────────────────────────────────────────────────
    {
        int total = population * n;
        int BS = 256, GS = (total + BS - 1) / BS;
        init_rng_kernel<<<GS, BS>>>(gpu.d_rng, (unsigned long long)seed, total);
        CUDA_CHECK(cudaGetLastError());
    }

    // ── Precompute base_theta_table ───────────────────────────────────────────
    {
        vector<double> btheta(population / 2);
        for (int r = 0; r < population / 2; ++r)
            btheta[r] = (current_theta * M_PI) / (double)(r + 1);
        CUDA_CHECK(cudaMemcpy(gpu.d_base_theta, btheta.data(),
                              btheta.size() * sizeof(double), cudaMemcpyHostToDevice));
    }

    // ── Initial neighbours + sort ─────────────────────────────────────────────
    {
        int total = population * n, BS = 256, GS = (total + BS - 1) / BS;
        gen_neighbours_kernel<<<GS, BS>>>(gpu.d_q, n, gpu.d_neigh, population, gpu.d_rng);
        CUDA_CHECK(cudaGetLastError());
        int EBS = 256;
        energy_kernel<<<population, EBS, (size_t)EBS * sizeof(double)>>>(
            gpu.d_neigh, population, n, gpu.d_Q, gpu.d_energy);
        CUDA_CHECK(cudaGetLastError());
        init_indices_kernel<<<(population + 255) / 256, 256>>>(gpu.d_idx, population);
        CUDA_CHECK(cudaGetLastError());
        thrust::sort_by_key(thrust::device_ptr<double>(gpu.d_energy),
                            thrust::device_ptr<double>(gpu.d_energy) + population,
                            thrust::device_ptr<int>(gpu.d_idx));
    }

    // ── Fetch initial global best ─────────────────────────────────────────────
    int best_idx_h = 0;
    CUDA_CHECK(cudaMemcpy(&best_idx_h, gpu.d_idx, sizeof(int), cudaMemcpyDeviceToHost));
    vector<unsigned char> best_bits(n);
    CUDA_CHECK(cudaMemcpy(best_bits.data(),
                          gpu.d_neigh + (size_t)best_idx_h * n,
                          (size_t)n * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    vector<int> global_best_sol(n);
    for (int i = 0; i < n; ++i) global_best_sol[i] = (int)best_bits[i];

    double global_best_energy = 0.0;
    CUDA_CHECK(cudaMemcpy(&global_best_energy, gpu.d_energy,
                          sizeof(double), cudaMemcpyDeviceToHost));

    // ── Constants ────────────────────────────────────────────────────────────
    const double ENTROPY_THRESHOLD = 0.02;
    int record_interval = max(1, iterations / 100);
    vector<HistoryPoint> history;
    history.reserve(110);

    auto t_start = chrono::high_resolution_clock::now();

    // ── AEQTS main loop ───────────────────────────────────────────────────────
    for (int it = 0; it < iterations; ++it) {

        // Copy Q-bits to host for entropy calculation
        CUDA_CHECK(cudaMemcpy(qflat.data(), gpu.d_q,
                              qflat.size() * sizeof(double), cudaMemcpyDeviceToHost));
        double current_entropy = calc_entropy(qflat);

        // Generate neighbours
        {
            int total = population * n, BS = 256, GS = (total + BS - 1) / BS;
            gen_neighbours_kernel<<<GS, BS>>>(gpu.d_q, n, gpu.d_neigh, population, gpu.d_rng);
            CUDA_CHECK(cudaGetLastError());
        }

        // Compute energies + sort by energy asc
        {
            int EBS = 256;
            energy_kernel<<<population, EBS, (size_t)EBS * sizeof(double)>>>(
                gpu.d_neigh, population, n, gpu.d_Q, gpu.d_energy);
            CUDA_CHECK(cudaGetLastError());
            init_indices_kernel<<<(population + 255) / 256, 256>>>(gpu.d_idx, population);
            CUDA_CHECK(cudaGetLastError());
            thrust::sort_by_key(thrust::device_ptr<double>(gpu.d_energy),
                                thrust::device_ptr<double>(gpu.d_energy) + population,
                                thrust::device_ptr<int>(gpu.d_idx));
        }

        // Read best neighbour this iteration
        int    cur_best_idx    = 0;
        double cur_best_energy = 0.0;
        CUDA_CHECK(cudaMemcpy(&cur_best_idx,    gpu.d_idx,    sizeof(int),    cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&cur_best_energy, gpu.d_energy, sizeof(double), cudaMemcpyDeviceToHost));

        // Update global best
        if (cur_best_energy < global_best_energy) {
            global_best_energy = cur_best_energy;
            CUDA_CHECK(cudaMemcpy(best_bits.data(),
                                  gpu.d_neigh + (size_t)cur_best_idx * n,
                                  (size_t)n * sizeof(unsigned char), cudaMemcpyDeviceToHost));
            for (int i = 0; i < n; ++i) global_best_sol[i] = (int)best_bits[i];
        }

        // Quantum rotation update
        updateQ_pairs_kernel<<<(n + 255) / 256, 256>>>(
            gpu.d_neigh, n, population, gpu.d_idx, gpu.d_base_theta, gpu.d_q);
        CUDA_CHECK(cudaGetLastError());

        // Record history checkpoint
        if (it % record_interval == 0) {
            double obj = 0.0, tw = 0.0;
            for (int i = 0; i < n; ++i) {
                obj += global_best_sol[i] * values[i];
                tw  += global_best_sol[i] * weights[i];
            }
            history.push_back({it, global_best_energy, cur_best_energy,
                                obj, current_entropy, tw <= capacity});
        }

        // Early stop on entropy collapse
        if (current_entropy <= ENTROPY_THRESHOLD) {
            cerr << "[aeqts] Early stop at iter " << it
                 << " entropy=" << current_entropy << "\n";
            break;
        }
    }

    // ── Final state ───────────────────────────────────────────────────────────
    CUDA_CHECK(cudaMemcpy(qflat.data(), gpu.d_q,
                          qflat.size() * sizeof(double), cudaMemcpyDeviceToHost));
    double final_entropy = calc_entropy(qflat);
    double final_obj = 0.0, final_weight = 0.0;
    for (int i = 0; i < n; ++i) {
        final_obj    += global_best_sol[i] * values[i];
        final_weight += global_best_sol[i] * weights[i];
    }
    bool final_feasible = (final_weight <= capacity);

    // Final history point
    history.push_back({iterations, global_best_energy, global_best_energy,
                       final_obj, final_entropy, final_feasible});

    double elapsed_ms = chrono::duration<double, milli>(
        chrono::high_resolution_clock::now() - t_start).count();

    // ── JSON output to stdout ─────────────────────────────────────────────────
    cout << "{\n"
         << "  \"solution\": [";
    for (int i = 0; i < n; ++i) {
        cout << global_best_sol[i];
        if (i < n - 1) cout << ",";
    }
    cout << "],\n"
         << "  \"energy\": "              << jdbl(global_best_energy) << ",\n"
         << "  \"total_value\": "         << jdbl(final_obj)          << ",\n"
         << "  \"total_weight\": "        << jdbl(final_weight)       << ",\n"
         << "  \"is_feasible\": "         << jbool(final_feasible)    << ",\n"
         << "  \"computation_time_ms\": " << jdbl(elapsed_ms)         << ",\n"
         << "  \"device\": \"cuda\",\n"
         << "  \"history\": [\n";

    for (size_t i = 0; i < history.size(); ++i) {
        const auto& h = history[i];
        cout << "    {"
             << "\"iteration\":"      << h.iteration          << ","
             << "\"energy\":"         << jdbl(h.energy)       << ","
             << "\"current_energy\":" << jdbl(h.current_energy) << ","
             << "\"objective\":"      << jdbl(h.objective)    << ","
             << "\"entropy\":"        << jdbl(h.entropy)      << ","
             << "\"is_feasible\":"    << jbool(h.is_feasible) << "}";
        if (i < history.size() - 1) cout << ",";
        cout << "\n";
    }
    cout << "  ]\n"
         << "}\n";

    return 0;
}

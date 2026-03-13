// aeqts.cu — AEQTS CUDA solver for Knapsack (Streaming JSON Lines)
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

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

using namespace std;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ── CUDA error check ──
#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        std::cerr << "[CUDA ERROR] " << __FILE__ << ":" << __LINE__ \
                  << " : " << cudaGetErrorString(_e) << std::endl; \
        std::exit(1); \
    } \
} while(0)

static mt19937 g_rng;

// ── CUDA Kernels ──
__global__ void init_rng_kernel(curandStatePhilox4_32_10_t* st, unsigned long long seed, int total) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid < total) curand_init(seed, (unsigned long long)tid, 0ULL, &st[tid]);
}

__global__ void init_indices_kernel(int* idx, int N) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid < N) idx[tid] = tid;
}

__global__ void gen_neighbours_kernel(const double* __restrict__ q, int n_items, unsigned char* __restrict__ neighbours, int N, curandStatePhilox4_32_10_t* __restrict__ rng) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= N * n_items) return;
    int i = tid % n_items;
    double beta = q[2 * i + 1];
    double prob = beta * beta;
    float  u    = curand_uniform(&rng[tid]);
    neighbours[tid] = (u <= (float)prob) ? 1 : 0;
}

__global__ void energy_kernel(const unsigned char* __restrict__ neighbours, int N, int n_items, const double* __restrict__ Q, double* __restrict__ energies) {
    int n = (int)blockIdx.x;
    if (n >= N) return;
    const unsigned char* x = neighbours + n * n_items;
    double local = 0.0;
    for (int i = (int)threadIdx.x; i < n_items; i += (int)blockDim.x) {
        if (!x[i]) continue;
        const double* Qi = Q + i * n_items;
        double rowSum = 0.0;
        for (int j = 0; j < n_items; ++j) if (x[j]) rowSum += Qi[j];
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

__global__ void updateQ_pairs_kernel(const unsigned char* __restrict__ neighbours, int n_items, int N, const int* __restrict__ sorted_idx, const double* __restrict__ base_theta_table, double* __restrict__ q) {
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
    double alpha = q[2 * i + 0], beta = q[2 * i + 1];
    double sign_mask = (alpha * beta < 0.0) ? -1.0 : 1.0;
    double final_theta = raw * sign_mask;
    double s, c;
    sincos(final_theta, &s, &c);
    q[2 * i + 0] = alpha * c - beta * s;
    q[2 * i + 1] = alpha * s + beta * c;
}

// ── GPU Context ──
struct GPUContext {
    int n_items = 0, N = 0;
    double *d_Q = nullptr, *d_q = nullptr, *d_energy = nullptr, *d_base_theta = nullptr;
    unsigned char* d_neigh = nullptr; int* d_idx = nullptr; curandStatePhilox4_32_10_t* d_rng = nullptr;
    void free_all() {
        if (d_Q) cudaFree(d_Q); if (d_q) cudaFree(d_q); if (d_neigh) cudaFree(d_neigh);
        if (d_energy) cudaFree(d_energy); if (d_idx) cudaFree(d_idx); if (d_base_theta) cudaFree(d_base_theta);
        if (d_rng) cudaFree(d_rng);
    }
    ~GPUContext() { free_all(); }
    void init(int n_items_, int N_) {
        free_all(); n_items = n_items_; N = N_;
        CUDA_CHECK(cudaMalloc(&d_Q, (size_t)n_items * n_items * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_q, (size_t)2 * n_items * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_neigh, (size_t)N * n_items * sizeof(unsigned char)));
        CUDA_CHECK(cudaMalloc(&d_energy, (size_t)N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_idx, (size_t)N * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_base_theta, (size_t)(N / 2) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_rng, (size_t)N * n_items * sizeof(curandStatePhilox4_32_10_t)));
    }
};

double calc_entropy(const vector<double>& qflat) {
    double total = 0.0; int n = (int)qflat.size() / 2;
    for (int i = 0; i < n; ++i) {
        double a = qflat[2 * i], b = qflat[2 * i + 1];
        double p0 = a * a, p1 = b * b;
        if (p0 > 1e-15 && p1 > 1e-15) total += -(p0 * log2(p0) + p1 * log2(p1));
    }
    return total / n;
}

vector<double> parse_csv(const string& s) {
    vector<double> v; istringstream ss(s); string tok;
    while (getline(ss, tok, ',')) if (!tok.empty()) v.push_back(stod(tok));
    return v;
}

static string jdbl(double d) {
    ostringstream os; os << setprecision(6) << d; return os.str();
}

// ── Main ──
int main(int argc, char* argv[]) {
    vector<double> weights, values;
    double capacity = 0.0, penalty = 10.0, timeout = 30.0;
    int population = 50, iterations = 1000; long long seed = -1LL;

    for (int i = 1; i < argc - 1; ++i) {
        string key = argv[i], val = argv[i + 1];
        if (key == "--weights") { weights = parse_csv(val); ++i; }
        else if (key == "--values") { values = parse_csv(val); ++i; }
        else if (key == "--capacity") { capacity = stod(val); ++i; }
        else if (key == "--penalty") { penalty = stod(val); ++i; }
        else if (key == "--population") { population = stoi(val); ++i; }
        else if (key == "--iterations") { iterations = stoi(val); ++i; }
        else if (key == "--timeout") { timeout = stod(val); ++i; }
        else if (key == "--seed") { seed = stoll(val); ++i; }
    }

    int n_items = weights.size();
    if (n_items == 0 || values.size() != n_items) {
        cerr << "[aeqts] Error: Invalid weights/values\n"; return 1;
    }
    if (seed < 0LL) seed = (long long)chrono::high_resolution_clock::now().time_since_epoch().count();
    g_rng.seed((unsigned long long)seed);

    // ── Build QUBO Matrix (Knapsack with Slack bits) ──
    int K = capacity > 0 ? max(1, (int)ceil(log2(capacity + 1.0))) : 1;
    int total_vars = n_items + K;
    vector<double> coeffs(total_vars);
    for (int i = 0; i < n_items; ++i) coeffs[i] = weights[i];
    for (int k = 0; k < K; ++k) coeffs[n_items + k] = pow(2.0, k);

    vector<double> Qflat(total_vars * total_vars, 0.0);
    for (int i = 0; i < total_vars; ++i) {
        double v_i = (i < n_items) ? values[i] : 0.0;
        Qflat[i * total_vars + i] = penalty * (coeffs[i] * coeffs[i] - 2.0 * capacity * coeffs[i]) - v_i;
        for (int j = i + 1; j < total_vars; ++j) {
            double val = 2.0 * penalty * coeffs[i] * coeffs[j];
            Qflat[i * total_vars + j] = val / 2.0;
            Qflat[j * total_vars + i] = val / 2.0;
        }
    }

    GPUContext gpu; gpu.init(total_vars, population);
    CUDA_CHECK(cudaMemcpy(gpu.d_Q, Qflat.data(), total_vars * total_vars * sizeof(double), cudaMemcpyHostToDevice));

    vector<double> theta_list; for (int i = 1; i <= 10; ++i) theta_list.push_back(i * 0.01);
    uniform_int_distribution<> tdist(0, (int)theta_list.size() - 1);
    double current_theta = theta_list[tdist(g_rng)];

    vector<double> qflat(2 * total_vars, 1.0 / sqrt(2.0));
    CUDA_CHECK(cudaMemcpy(gpu.d_q, qflat.data(), qflat.size() * sizeof(double), cudaMemcpyHostToDevice));

    int total_rng = population * total_vars, BS = 256, GS = (total_rng + BS - 1) / BS;
    init_rng_kernel<<<GS, BS>>>(gpu.d_rng, (unsigned long long)seed, total_rng);
    CUDA_CHECK(cudaGetLastError());

    vector<double> btheta(population / 2);
    for (int r = 0; r < population / 2; ++r) btheta[r] = (current_theta * M_PI) / (double)(r + 1);
    CUDA_CHECK(cudaMemcpy(gpu.d_base_theta, btheta.data(), btheta.size() * sizeof(double), cudaMemcpyHostToDevice));

    double global_best_energy = 1e15;
    vector<int> global_best_sol(total_vars, 0);
    vector<unsigned char> best_bits(total_vars);
    int record_interval = max(1, iterations / 100);
    
    auto t_start = chrono::high_resolution_clock::now();

    // ── AEQTS Loop ──
    for (int it = 0; it < iterations; ++it) {
        double elapsed_sec = chrono::duration<double>(chrono::high_resolution_clock::now() - t_start).count();
        if (elapsed_sec > timeout) break;

        CUDA_CHECK(cudaMemcpy(qflat.data(), gpu.d_q, qflat.size() * sizeof(double), cudaMemcpyDeviceToHost));
        double current_entropy = calc_entropy(qflat);

        gen_neighbours_kernel<<<GS, BS>>>(gpu.d_q, total_vars, gpu.d_neigh, population, gpu.d_rng);
        energy_kernel<<<population, 256, 256 * sizeof(double)>>>(gpu.d_neigh, population, total_vars, gpu.d_Q, gpu.d_energy);
        init_indices_kernel<<<(population + 255) / 256, 256>>>(gpu.d_idx, population);
        thrust::sort_by_key(thrust::device_ptr<double>(gpu.d_energy), thrust::device_ptr<double>(gpu.d_energy) + population, thrust::device_ptr<int>(gpu.d_idx));

        int cur_best_idx = 0; double cur_best_energy = 0.0;
        CUDA_CHECK(cudaMemcpy(&cur_best_idx, gpu.d_idx, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&cur_best_energy, gpu.d_energy, sizeof(double), cudaMemcpyDeviceToHost));

        if (cur_best_energy < global_best_energy) {
            global_best_energy = cur_best_energy;
            CUDA_CHECK(cudaMemcpy(best_bits.data(), gpu.d_neigh + cur_best_idx * total_vars, total_vars * sizeof(unsigned char), cudaMemcpyDeviceToHost));
            for (int i = 0; i < total_vars; ++i) global_best_sol[i] = (int)best_bits[i];
        }

        updateQ_pairs_kernel<<<(total_vars + 255) / 256, 256>>>(gpu.d_neigh, total_vars, population, gpu.d_idx, gpu.d_base_theta, gpu.d_q);

        // 💡 即時輸出 (Streaming) JSON Line
        if (it % record_interval == 0 || it == iterations - 1) {
            double obj_val = 0.0, weight_sum = 0.0;
            for(int i=0; i<n_items; ++i) {
                if (global_best_sol[i]) { obj_val += values[i]; weight_sum += weights[i]; }
            }
            bool is_feasible = (weight_sum <= capacity);
            
            ostringstream probs_json;
            for(int i=0; i<total_vars; ++i) {
                probs_json << jdbl(qflat[2*i+1] * qflat[2*i+1]);
                if (i < total_vars - 1) probs_json << ",";
            }

            cout << "{\"type\":\"progress\", \"iteration\":" << it 
                 << ", \"energy\":" << jdbl(global_best_energy) 
                 << ", \"current_energy\":" << jdbl(cur_best_energy)
                 << ", \"objective\":" << jdbl(obj_val)
                 << ", \"entropy\":" << jdbl(current_entropy)
                 << ", \"is_feasible\":" << (is_feasible ? "true" : "false")
                 << ", \"qubit_probs\":[" << probs_json.str() << "]}" << endl;
        }
        if (current_entropy <= 0.02) break;
    }

    double elapsed_ms = chrono::duration<double, milli>(chrono::high_resolution_clock::now() - t_start).count();

    // ── JSON Output (Final) ──
    cout << "{\"type\":\"final\", \"solution\": [";
    for (int i = 0; i < total_vars; ++i) {
        cout << global_best_sol[i];
        if (i < total_vars - 1) cout << ",";
    }
    cout << "], \"energy\": " << jdbl(global_best_energy)
         << ", \"computation_time_ms\": " << jdbl(elapsed_ms)
         << ", \"device\": \"cuda\"}\n";
    return 0;
}
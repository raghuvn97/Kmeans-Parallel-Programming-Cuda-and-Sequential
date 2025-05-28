#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <limits>
#include <fstream>
#include <iomanip>
#include <cfloat>

using namespace std;
using namespace std::chrono;

__global__ void assign_clusters(const float* data, const float* centroids, int* assignments, float* sse, int N, int D, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const float* point = &data[idx * D];
    float min_dist = FLT_MAX;
    int best_k = 0;

    for (int k = 0; k < K; k++) {
        const float* centroid = &centroids[k * D];
        float dist = 0.0f;
        for (int d = 0; d < D; d++) {
            float diff = point[d] - centroid[d];
            dist += diff * diff;
        }
        if (dist < min_dist) {
            min_dist = dist;
            best_k = k;
        }
    }
    assignments[idx] = best_k;
    atomicAdd(sse, min_dist);  // accumulate SSE
}

__global__ void compute_new_centroids(const float* data, const int* assignments, float* new_centroids, int* counts, int N, int D, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int cluster_id = assignments[idx];
    for (int d = 0; d < D; d++) {
        atomicAdd(&new_centroids[cluster_id * D + d], data[idx * D + d]);
    }
    atomicAdd(&counts[cluster_id], 1);
}

void checkCuda(cudaError_t result, const char* msg) {
    if (result != cudaSuccess) {
        cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(result) << endl;
        exit(1);
    }
}

vector<float> load_binary_data(const string& filename, size_t& N, size_t& D) {
    ifstream file(filename, ios::binary | ios::ate);
    if (!file) {
        cerr << "Error opening file: " << filename << endl;
        exit(1);
    }
    streamsize size = file.tellg();
    file.seekg(0, ios::beg);
    vector<float> data(size / sizeof(float));
    file.read(reinterpret_cast<char*>(data.data()), size);
    return data;
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        cerr << "ArgsNeeded: " << argv[0] << " dataFile.bin N D K\n";
        return 1;
    }

    const string filename = argv[1];
    const int N = stoi(argv[2]);
    const int D = stoi(argv[3]);
    const int K = stoi(argv[4]);
    const int max_iter = 20;

    size_t actual_N = N, actual_D = D;
    vector<float> h_data = load_binary_data(filename, actual_N, actual_D);
    if (N * D != static_cast<int>(h_data.size())) {
        cerr << "Invalid dimensions!" << endl;
        return 1;
    }

    vector<float> h_centroids(K * D);
    vector<int> h_assignments(N);
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, N - 1);

    // K-Means++ Initialization
    int first_idx = dis(gen);
    for (int d = 0; d < D; d++)
        h_centroids[d] = h_data[first_idx * D + d];

    for (int k = 1; k < K; k++) {
        vector<float> min_distances(N, FLT_MAX);
        float total = 0;
        for (int i = 0; i < N; i++) {
            float dist = 0.0f;
            for (int d = 0; d < D; d++) {
                float diff = h_data[i * D + d] - h_centroids[(k - 1) * D + d];
                dist += diff * diff;
            }
            min_distances[i] = min(min_distances[i], dist);
            total += min_distances[i];
        }
        float threshold = uniform_real_distribution<float>(0.0, total)(gen);
        float cumsum = 0.0;
        int idx = 0;
        while (cumsum < threshold && idx < N - 1) {
            cumsum += min_distances[idx++];
        }
        for (int d = 0; d < D; d++)
            h_centroids[k * D + d] = h_data[idx * D + d];
    }

    float *d_data, *d_centroids, *d_new_centroids, *d_sse;
    int *d_assignments, *d_counts;
    size_t data_size = N * D * sizeof(float);
    size_t centroid_size = K * D * sizeof(float);
    size_t assignment_size = N * sizeof(int);
    size_t count_size = K * sizeof(int);

    checkCuda(cudaMalloc(&d_data, data_size), "data");
    checkCuda(cudaMalloc(&d_centroids, centroid_size), "centroids");
    checkCuda(cudaMalloc(&d_new_centroids, centroid_size), "new_centroids");
    checkCuda(cudaMalloc(&d_assignments, assignment_size), "assignments");
    checkCuda(cudaMalloc(&d_counts, count_size), "counts");
    checkCuda(cudaMalloc(&d_sse, sizeof(float)), "sse");

    checkCuda(cudaMemcpy(d_data, h_data.data(), data_size, cudaMemcpyHostToDevice), "copy data");

    auto start = high_resolution_clock::now();

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    for (int iter = 0; iter < max_iter; iter++) {
        checkCuda(cudaMemcpy(d_centroids, h_centroids.data(), centroid_size, cudaMemcpyHostToDevice), "copy centroids");

        checkCuda(cudaMemset(d_sse, 0, sizeof(float)), "memset sse");
        assign_clusters<<<blocks, threads>>>(d_data, d_centroids, d_assignments, d_sse, N, D, K);
        checkCuda(cudaDeviceSynchronize(), "assign_clusters");

        checkCuda(cudaMemset(d_new_centroids, 0, centroid_size), "memset new_centroids");
        checkCuda(cudaMemset(d_counts, 0, count_size), "memset counts");

        compute_new_centroids<<<blocks, threads>>>(d_data, d_assignments, d_new_centroids, d_counts, N, D, K);
        checkCuda(cudaDeviceSynchronize(), "compute_new_centroids");

        vector<float> new_centroids(K * D);
        vector<int> counts(K);
        checkCuda(cudaMemcpy(new_centroids.data(), d_new_centroids, centroid_size, cudaMemcpyDeviceToHost), "copy back centroids");
        checkCuda(cudaMemcpy(counts.data(), d_counts, count_size, cudaMemcpyDeviceToHost), "copy back counts");

        for (int k = 0; k < K; k++) {
            if (counts[k] > 0) {
                for (int d = 0; d < D; d++) {
                    h_centroids[k * D + d] = new_centroids[k * D + d] / counts[k];
                }
            }
        }

        float h_sse;
        checkCuda(cudaMemcpy(&h_sse, d_sse, sizeof(float), cudaMemcpyDeviceToHost), "copy sse");
        cout << "Iteration " << (iter + 1) << " - SSE: " << fixed << setprecision(2) << h_sse << endl;
    }

    auto end = high_resolution_clock::now();
    cout << "\nClustering completed in " << duration_cast<milliseconds>(end - start).count() << " ms\n";
    cout << "\nFinal cluster centroids:\n";

    // Get final assignments to compute cluster sizes
    checkCuda(cudaMemcpy(h_assignments.data(), d_assignments, assignment_size, cudaMemcpyDeviceToHost), "copy back assignments");
    vector<int> cluster_sizes(K, 0);
    for (int i = 0; i < N; i++) {
        cluster_sizes[h_assignments[i]]++;
    }

    for (int k = 0; k < K; k++) {
        cout << "Cluster " << k << " (size " << cluster_sizes[k] << "): [";
        for (int d = 0; d < D; d++) {
            cout << fixed << setprecision(4) << h_centroids[k * D + d];
            if (d < D - 1) cout << ", ";
        }
        cout << "]\n";
    }

    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_new_centroids);
    cudaFree(d_assignments);
    cudaFree(d_counts);
    cudaFree(d_sse);

    return 0;
}

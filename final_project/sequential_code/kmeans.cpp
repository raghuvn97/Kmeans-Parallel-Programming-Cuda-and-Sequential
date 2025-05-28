#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <random>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <iomanip>

using namespace std;
using namespace std::chrono;

// Euclidean distance between two points
float distance(const float* a, const float* b, int D) {
    float sum = 0.0f;
    for (int i = 0; i < D; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

// K-Means clustering implementation
void kmeans(const float* data, int N, int D, int K, int max_iter, 
            vector<vector<float>>& centroids, vector<int>& assignments) {
    
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, N-1);

    // K-means++ initialization
    centroids.resize(K, vector<float>(D));
    vector<float> min_distances(N, numeric_limits<float>::max());
    
    // Select First centroid random
    int first_idx = dis(gen);
    for (int d = 0; d < D; d++) {
        centroids[0][d] = data[first_idx * D + d];
    }

    // Subsequent centroids
    for (int k = 1; k < K; k++) {
        // Update minimum distances
        float total_distance = 0.0f;
        for (int i = 0; i < N; i++) {
            float dist = distance(&data[i * D], &centroids[k-1][0], D);
            if (dist < min_distances[i]) {
                min_distances[i] = dist;
            }
            total_distance += min_distances[i];
        }
        
        // Select next centroid
        uniform_real_distribution<float> dist_dist(0.0, total_distance);
        float threshold = dist_dist(gen);
        float running_sum = 0.0f;
        int selected_idx = 0;
        while (running_sum < threshold && selected_idx < N-1) {
            running_sum += min_distances[selected_idx];
            selected_idx++;
        }
        
        for (int d = 0; d < D; d++) {
            centroids[k][d] = data[selected_idx * D + d];
        }
    }

    assignments.resize(N);
    vector<int> counts(K, 0);
    vector<vector<float>> new_centroids(K, vector<float>(D, 0.0f));

    for (int iter = 0; iter < max_iter; iter++) {
        // Assignment step
        float total_sse = 0.0f;
        fill(counts.begin(), counts.end(), 0);
        for (auto& centroid : new_centroids) {
            fill(centroid.begin(), centroid.end(), 0.0f);
        }

        for (int i = 0; i < N; i++) {
            float min_dist = numeric_limits<float>::max();
            int best_k = 0;
            const float* point = &data[i * D];
            
            for (int k = 0; k < K; k++) {
                float dist = distance(point, &centroids[k][0], D);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_k = k;
                }
            }
            
            assignments[i] = best_k;
            counts[best_k]++;
            total_sse += min_dist;
            
            for (int d = 0; d < D; d++) {
                new_centroids[best_k][d] += point[d];
            }
        }

        // Update centroids
        bool converged = true;
        for (int k = 0; k < K; k++) {
            if (counts[k] == 0) {
                // Reinitialize empty cluster
                int random_idx = dis(gen);
                for (int d = 0; d < D; d++) {
                    new_centroids[k][d] = data[random_idx * D + d];
                }
                converged = false;
                continue;
            }
            
            for (int d = 0; d < D; d++) {
                new_centroids[k][d] /= counts[k];
                if (fabs(new_centroids[k][d] - centroids[k][d]) > 1e-6) {
                    converged = false;
                }
                centroids[k][d] = new_centroids[k][d];
            }
        }

        cout << "Iteration " << iter + 1 << " - SSE: " << fixed << setprecision(2) << total_sse << endl;
        if (converged) break;
    }
}

// Load binary data file
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
        cerr << "  N: Number of points\n";
        cerr << "  D: Dimensions per point\n";
        cerr << "  K: Number of clusters\n";
        return 1;
    }

    const string filename = argv[1];
    const int N = stoi(argv[2]);
    const int D = stoi(argv[3]);
    const int K = stoi(argv[4]);
    const int max_iter = 20;

    // Load data
    cout << "Loading data from " << filename << "..." << endl;
    size_t actual_N = N, actual_D = D;
    vector<float> data = load_binary_data(filename, actual_N, actual_D);
    
    if (N * D != static_cast<int>(data.size())) {
        cerr << "Error: File size doesn't match expected dimensions\n";
        cerr << "Expected " << N * D << " elements (" << N << " points × " << D << " dimensions)\n";
        cerr << "Actual file contains " << data.size() << " elements\n";
        return 1;
    }

    // Run K-Means
    vector<vector<float>> centroids;
    vector<int> assignments;
    
    cout << "\nRunning K-Means clustering (N=" << N << ", D=" << D << ", K=" << K << ")..." << endl;
    auto start = high_resolution_clock::now();
    kmeans(data.data(), N, D, K, max_iter, centroids, assignments);
    auto end = high_resolution_clock::now();

    // Calculate cluster sizes
    vector<int> cluster_sizes(K, 0);
    for (int cluster : assignments) {
        cluster_sizes[cluster]++;
    }

    // Print results
    cout << "\nClustering completed in " << duration_cast<milliseconds>(end - start).count() << " ms\n";
    cout << "\nFinal cluster centroids:\n";
    for (int k = 0; k < K; k++) {
        cout << "Cluster " << k << " (size " << cluster_sizes[k] << "): [";
        for (int d = 0; d < D; d++) {
            cout << fixed << setprecision(4) << centroids[k][d];
            if (d < D - 1) cout << ", ";
        }
        cout << "]" << endl;
    }

    return 0;
}
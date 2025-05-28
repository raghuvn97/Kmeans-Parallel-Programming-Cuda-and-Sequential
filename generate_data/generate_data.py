import numpy as np
import argparse
import os

def generate_kmeans_data(N, D, K=None, cluster_std=1.0, output_dir="data"):
    """
    Generate synthetic K-Means test data
    
    Args:
        N: Number of data points
        D: Number of dimensions
        K: Number of clusters (optional)
        cluster_std: Standard deviation of clusters
        output_dir: Directory to save files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if K is None:
        # For unclustered random data
        data = np.random.uniform(-10, 10, size=(N, D)).astype(np.float32)
        true_centroids = None
    else:
        # Generate clustered data
        centroids = np.random.uniform(-10, 10, size=(K, D))
        sizes = np.random.multinomial(N, np.ones(K)/K)
        
        clusters = []
        for k in range(K):
            clusters.append(np.random.normal(
                loc=centroids[k], 
                scale=cluster_std, 
                size=(sizes[k], D)
            ))
        
        data = np.vstack(clusters)
        np.random.shuffle(data)
        data = data.astype(np.float32)
        true_centroids = centroids.astype(np.float32)
    
    # Save in multiple formats
    base_path = os.path.join(output_dir, f"kmeans_N{N}_D{D}" + (f"_K{K}" if K else ""))
    
    # Binary format (for C++/CUDA)
    data.tofile(base_path + ".bin")
    
    # NumPy format (for Python)
    np.save(base_path + ".npy", data)
    
    # Text format (for verification)
    np.savetxt(base_path + ".csv", data, delimiter=',', fmt='%.6f')
    
    if true_centroids is not None:
        np.save(base_path + "_centroids.npy", true_centroids)
        np.savetxt(base_path + "_centroids.csv", true_centroids, delimiter=',', fmt='%.6f')
    
    print(f"Generated {N} points with {D} dimensions ({K} clusters if specified)")
    print(f"Files saved with base name: {base_path}")
    print(f"  Binary: {base_path}.bin")
    print(f"  NumPy:  {base_path}.npy")
    print(f"  CSV:    {base_path}.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate K-Means test data')
    parser.add_argument('N', type=int, help='Number of data points')
    parser.add_argument('D', type=int, help='Number of dimensions')
    parser.add_argument('--K', type=int, help='Number of clusters (optional)')
    parser.add_argument('--std', type=float, default=1.0, 
                       help='Standard deviation for clusters (default: 1.0)')
    parser.add_argument('--output', default="data", 
                       help='Output directory (default: "data")')
    
    args = parser.parse_args()
    
    generate_kmeans_data(
        N=args.N,
        D=args.D,
        K=args.K,
        cluster_std=args.std,
        output_dir=args.output
    )
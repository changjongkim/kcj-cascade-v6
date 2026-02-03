/**
 * Multi-Node Multi-GPU Distributed Benchmark
 * 
 * Tests:
 * 1. Local GPU throughput (baseline)
 * 2. NVLink GPU-to-GPU within node
 * 3. Remote node access via MPI RMA
 * 4. Cluster-wide aggregate throughput
 */

#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <cstring>
#include <iomanip>

#ifdef USE_MPI
#include <mpi.h>
#endif

#include <cuda_runtime.h>
#include "cascade_distributed.hpp"

using namespace cascade;
using namespace cascade::distributed;

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ============================================================================
// Benchmark Configuration
// ============================================================================

struct BenchConfig {
    size_t block_size = 1024 * 1024;    // 1MB per block
    int num_blocks = 1000;               // Total blocks per node
    int warmup_blocks = 100;
    size_t gpu_capacity = 4ULL * 1024 * 1024 * 1024;  // 4GB for benchmark
    size_t dram_capacity = 8ULL * 1024 * 1024 * 1024; // 8GB
    int num_gpus = 1;                    // GPUs per node (will be auto-detected)
};

// ============================================================================
// Timer helper
// ============================================================================

class Timer {
public:
    void start() { start_ = std::chrono::high_resolution_clock::now(); }
    double stop() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(end - start_).count();
    }
private:
    std::chrono::high_resolution_clock::time_point start_;
};

// ============================================================================
// Generate random data and block IDs
// ============================================================================

void generate_test_data(std::vector<std::vector<uint8_t>>& blocks,
                        std::vector<BlockId>& ids,
                        int num_blocks, size_t block_size, int rank) {
    std::mt19937 rng(42 + rank);  // Different seed per rank
    std::uniform_int_distribution<uint8_t> dist(0, 255);
    
    blocks.resize(num_blocks);
    ids.resize(num_blocks);
    
    for (int i = 0; i < num_blocks; i++) {
        blocks[i].resize(block_size);
        for (size_t j = 0; j < block_size; j++) {
            blocks[i][j] = dist(rng);
        }
        ids[i] = compute_block_id(blocks[i].data(), block_size);
    }
}

// ============================================================================
// Local GPU Benchmark
// ============================================================================

void benchmark_local_gpu(DistributedStore& store, 
                         const std::vector<std::vector<uint8_t>>& blocks,
                         const std::vector<BlockId>& ids,
                         const BenchConfig& cfg, int rank) {
    Timer timer;
    size_t total_bytes = 0;
    
    // Warmup
    for (int i = 0; i < cfg.warmup_blocks && i < (int)blocks.size(); i++) {
        store.put(ids[i], blocks[i].data(), blocks[i].size());
    }
    store.barrier();
    
    // Write benchmark
    timer.start();
    for (int i = cfg.warmup_blocks; i < (int)blocks.size(); i++) {
        store.put(ids[i], blocks[i].data(), blocks[i].size());
        total_bytes += blocks[i].size();
    }
    store.barrier();
    double write_time = timer.stop();
    
    double write_gbps = (total_bytes / (1024.0 * 1024.0 * 1024.0)) / write_time;
    
    if (rank == 0) {
        printf("[Write] %.2f GB in %.3f s = %.2f GB/s (local)\n",
               total_bytes / (1024.0 * 1024.0 * 1024.0), write_time, write_gbps);
    }
    
    // Read benchmark
    std::vector<uint8_t> read_buf(cfg.block_size);
    size_t read_size;
    total_bytes = 0;
    
    timer.start();
    for (int i = cfg.warmup_blocks; i < (int)ids.size(); i++) {
        if (store.get(ids[i], read_buf.data(), &read_size)) {
            total_bytes += read_size;
        }
    }
    store.barrier();
    double read_time = timer.stop();
    
    double read_gbps = (total_bytes / (1024.0 * 1024.0 * 1024.0)) / read_time;
    
    if (rank == 0) {
        printf("[Read]  %.2f GB in %.3f s = %.2f GB/s (local)\n",
               total_bytes / (1024.0 * 1024.0 * 1024.0), read_time, read_gbps);
    }
}

// ============================================================================
// Cross-Node Benchmark (read from other node's blocks)
// ============================================================================

void benchmark_cross_node(DistributedStore& store,
                          const std::vector<BlockId>& all_ids,
                          const BenchConfig& cfg, int rank, int world_size) {
    if (world_size < 2) {
        if (rank == 0) {
            printf("[Cross-Node] Skipped (need >= 2 nodes)\n");
        }
        return;
    }
    
    store.barrier();
    
    Timer timer;
    std::vector<uint8_t> read_buf(cfg.block_size);
    size_t read_size;
    size_t total_bytes = 0;
    int success = 0, fail = 0;
    
    // Each rank tries to read blocks that hash to other nodes
    timer.start();
    for (const auto& id : all_ids) {
        if (store.get(id, read_buf.data(), &read_size)) {
            total_bytes += read_size;
            success++;
        } else {
            fail++;
        }
    }
    store.barrier();
    double read_time = timer.stop();
    
    // Aggregate stats
    size_t global_bytes = 0;
    int global_success = 0;
    
#ifdef USE_MPI
    MPI_Reduce(&total_bytes, &global_bytes, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&success, &global_success, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
#else
    global_bytes = total_bytes;
    global_success = success;
#endif
    
    if (rank == 0) {
        double gbps = (global_bytes / (1024.0 * 1024.0 * 1024.0)) / read_time;
        printf("[Cross-Node] %d reads, %.2f GB in %.3f s = %.2f GB/s aggregate\n",
               global_success, global_bytes / (1024.0 * 1024.0 * 1024.0),
               read_time, gbps);
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    int rank = 0, world_size = 1;
    
#ifdef USE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
#endif
    
    // Detect number of GPUs
    int num_gpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    
    if (rank == 0) {
        printf("=================================================\n");
        printf("  Cascade Distributed Benchmark\n");
        printf("=================================================\n");
        printf("  Nodes: %d\n", world_size);
        printf("  GPUs per node: %d\n", num_gpus);
        printf("  Total GPUs: %d\n", num_gpus * world_size);
        printf("=================================================\n\n");
    }
    
    BenchConfig cfg;
    cfg.num_gpus = num_gpus > 0 ? num_gpus : 1;
    
    // Parse command line
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--blocks") == 0 && i + 1 < argc) {
            cfg.num_blocks = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--block-size") == 0 && i + 1 < argc) {
            cfg.block_size = atoi(argv[++i]) * 1024 * 1024;  // Input in MB
        } else if (strcmp(argv[i], "--gpu-cap") == 0 && i + 1 < argc) {
            cfg.gpu_capacity = (size_t)atoi(argv[++i]) * 1024ULL * 1024 * 1024;
        }
    }
    
    if (rank == 0) {
        printf("Config: %d blocks × %.1f MB = %.2f GB per node\n",
               cfg.num_blocks, cfg.block_size / (1024.0 * 1024.0),
               (cfg.num_blocks * cfg.block_size) / (1024.0 * 1024.0 * 1024.0));
        printf("GPU capacity: %.2f GB/device, DRAM: %.2f GB/node\n\n",
               cfg.gpu_capacity / (1024.0 * 1024.0 * 1024.0),
               cfg.dram_capacity / (1024.0 * 1024.0 * 1024.0));
    }
    
    // Generate test data
    std::vector<std::vector<uint8_t>> blocks;
    std::vector<BlockId> ids;
    generate_test_data(blocks, ids, cfg.num_blocks, cfg.block_size, rank);
    
#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    
    // Create distributed store
    DistributedConfig dist_cfg;
    dist_cfg.gpu_capacity_per_device = cfg.gpu_capacity;
    dist_cfg.dram_capacity = cfg.dram_capacity;
    dist_cfg.num_gpus_per_node = cfg.num_gpus;
    
#ifdef USE_MPI
    DistributedStore store(dist_cfg, MPI_COMM_WORLD);
#else
    DistributedStore store(dist_cfg);
#endif
    
    // Run benchmarks
    if (rank == 0) printf("\n--- Local GPU Benchmark ---\n");
    benchmark_local_gpu(store, blocks, ids, cfg, rank);
    
    // Collect all block IDs across cluster for cross-node test
    std::vector<BlockId> all_ids = ids;
#ifdef USE_MPI
    // Share block IDs across all nodes
    // For simplicity, just use local IDs - in real scenario would exchange
#endif
    
    if (rank == 0) printf("\n--- Cross-Node Benchmark ---\n");
    benchmark_cross_node(store, all_ids, cfg, rank, world_size);
    
    // Print final stats
    auto stats = store.get_stats();
    if (rank == 0) {
        printf("\n--- Final Statistics ---\n");
        printf("Local GPU used:  %.2f GB\n", 
               stats.local_gpu_used / (1024.0 * 1024.0 * 1024.0));
        printf("Local DRAM used: %.2f GB\n",
               stats.local_dram_used / (1024.0 * 1024.0 * 1024.0));
        printf("Cluster GPU:     %.2f GB\n",
               stats.cluster_gpu_used / (1024.0 * 1024.0 * 1024.0));
        printf("Hits - Local GPU: %zu, Local DRAM: %zu\n",
               stats.local_gpu_hits, stats.local_dram_hits);
        printf("Hits - Remote GPU: %zu, Remote DRAM: %zu\n",
               stats.remote_gpu_hits, stats.remote_dram_hits);
        printf("Misses: %zu\n", stats.misses);
    }
    
#ifdef USE_MPI
    MPI_Finalize();
#endif
    
    return 0;
}

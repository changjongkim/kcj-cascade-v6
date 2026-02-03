/**
 * Cascade Distributed Backend - Multi-Node Multi-GPU KV Cache
 * 
 * GPU-aware MPI on HPE Slingshot (Perlmutter)
 * - MPICH_GPU_SUPPORT_ENABLED=1
 * - NCCL_NET_GDR_LEVEL=PHB
 * - GDRCopy v2.4
 * - 4x Slingshot NICs (cxi0-cxi3)
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <optional>
#include <thread>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <array>

#ifdef USE_MPI
#include <mpi.h>
#endif

#include "cascade.hpp"

namespace cascade {
namespace distributed {

// ============================================================================
// Block Location in Cluster
// ============================================================================

struct BlockLocation {
    int node_id;           // MPI rank
    int gpu_id;            // GPU device (-1 = DRAM)
    size_t offset;         // Offset in memory pool
    size_t size;           // Block size
    uint64_t timestamp;    // Creation time
    bool is_gpu;           // true=GPU, false=DRAM
    
    bool is_local(int my_rank) const { return node_id == my_rank; }
};

// ============================================================================
// Simple Sharded Index for Distributed Types (Self-contained)
// ============================================================================

template<typename V>
class DistributedIndex {
public:
    static constexpr size_t NUM_SHARDS = 256;
    
    DistributedIndex() = default;
    ~DistributedIndex() = default;
    
    bool put(const BlockId& key, V value, size_t size = 0) {
        size_t shard_id = std::hash<BlockId>{}(key) % NUM_SHARDS;
        auto& shard = shards_[shard_id];
        std::unique_lock lock(shard.mutex);
        shard.data[key] = value;
        return true;
    }
    
    std::optional<V> get(const BlockId& key) const {
        size_t shard_id = std::hash<BlockId>{}(key) % NUM_SHARDS;
        const auto& shard = shards_[shard_id];
        std::shared_lock lock(shard.mutex);
        auto it = shard.data.find(key);
        if (it == shard.data.end()) return std::nullopt;
        return it->second;
    }
    
    bool contains(const BlockId& key) const {
        return get(key).has_value();
    }
    
private:
    struct Shard {
        mutable std::shared_mutex mutex;
        std::unordered_map<BlockId, V> data;
    };
    std::array<Shard, NUM_SHARDS> shards_;
};

// ============================================================================
// Distributed Config
// ============================================================================

struct DistributedConfig {
    size_t gpu_capacity_per_device = 32ULL * 1024 * 1024 * 1024;
    size_t dram_capacity = 64ULL * 1024 * 1024 * 1024;
    int num_gpus_per_node = 4;
    size_t staging_buffer_size = 16 * 1024 * 1024;
    int num_staging_buffers = 32;
    bool sync_metadata = false;
};

// ============================================================================
// DRAM Backend with MPI RMA Window
// ============================================================================

struct DRAMBlock {
    size_t offset;
    size_t size;
};

class DistributedDRAMBackend {
public:
#ifdef USE_MPI
    DistributedDRAMBackend(size_t capacity, MPI_Comm comm);
#else
    explicit DistributedDRAMBackend(size_t capacity);
#endif
    ~DistributedDRAMBackend();
    
    bool put_local(const BlockId& id, const uint8_t* data, size_t size);
    bool get_local(const BlockId& id, uint8_t* out, size_t* out_size);
    bool get_remote(int target_rank, size_t offset, uint8_t* out, size_t size);
    
    size_t used_bytes() const { return used_.load(); }
    size_t capacity() const { return capacity_; }
    void barrier();
    
    DistributedIndex<DRAMBlock> index_;
    
private:
    size_t capacity_;
    void* dram_base_ = nullptr;
    std::atomic<size_t> write_offset_{0};
    std::atomic<size_t> used_{0};
    
#ifdef USE_MPI
    MPI_Comm comm_;
    MPI_Win window_;
    int rank_, world_size_;
#endif
};

// ============================================================================
// Multi-GPU Backend with NVLink + MPI
// ============================================================================

class DistributedGPUBackend {
public:
#ifdef USE_MPI
    DistributedGPUBackend(size_t cap_per_gpu, int num_gpus, MPI_Comm comm);
#else
    DistributedGPUBackend(size_t cap_per_gpu, int num_gpus);
#endif
    ~DistributedGPUBackend();
    
    bool put(const BlockId& id, const uint8_t* data, size_t size);
    bool get(const BlockId& id, uint8_t* out, size_t* out_size);
    bool put_local(const BlockId& id, const uint8_t* data, size_t size, int gpu);
    bool get_local(const BlockId& id, uint8_t* out, size_t* out_size);
    bool get_remote(int rank, size_t offset, size_t size, uint8_t* out);
    
    std::optional<BlockLocation> locate(const BlockId& id) const;
    size_t used_bytes() const;
    int get_target_node(const BlockId& id) const;
    int get_target_gpu(const BlockId& id) const;
    void sync_all();
    void barrier();
    
    // Global index
    DistributedIndex<BlockLocation> global_index_;
    
private:
    int num_gpus_;
    size_t cap_per_gpu_;
    int rank_ = 0, world_size_ = 1;
    
#ifdef USE_MPI
    MPI_Comm comm_;
    MPI_Win window_;
#endif
    
    std::vector<std::unique_ptr<GPUBackend>> gpus_;
    bool peer_[16][16] = {{false}};
    void* pinned_[32] = {nullptr};
    size_t staging_size_ = 16 * 1024 * 1024;
    
    void setup_nvlink();
    void init_window();
};

// ============================================================================
// Main Distributed Store
// ============================================================================

class DistributedStore {
public:
#ifdef USE_MPI
    DistributedStore(const DistributedConfig& cfg, MPI_Comm comm = MPI_COMM_WORLD);
#else
    explicit DistributedStore(const DistributedConfig& cfg);
#endif
    ~DistributedStore();
    
    bool put(const BlockId& id, const uint8_t* data, size_t size);
    bool get(const BlockId& id, uint8_t* out, size_t* out_size);
    bool contains(const BlockId& id) const;
    std::optional<BlockLocation> locate(const BlockId& id) const;
    
    size_t put_batch(const std::vector<BlockId>& ids,
                     const std::vector<const uint8_t*>& data,
                     const std::vector<size_t>& sizes);
    size_t get_batch(const std::vector<BlockId>& ids,
                     std::vector<uint8_t*>& out,
                     std::vector<size_t>& sizes);
    
    void barrier();
    void sync_metadata();
    
    struct Stats {
        size_t local_gpu_used, local_dram_used;
        size_t cluster_gpu_used, cluster_dram_used;
        size_t local_gpu_hits, local_dram_hits;
        size_t remote_gpu_hits, remote_dram_hits;
        size_t misses;
    };
    Stats get_stats();
    
    int rank() const { return rank_; }
    int world_size() const { return world_size_; }
    
private:
    DistributedConfig cfg_;
    int rank_, world_size_;
#ifdef USE_MPI
    MPI_Comm comm_;
#endif
    
    std::unique_ptr<DistributedGPUBackend> gpu_;
    std::unique_ptr<DistributedDRAMBackend> dram_;
    
    std::atomic<size_t> local_gpu_hits_{0}, local_dram_hits_{0};
    std::atomic<size_t> remote_gpu_hits_{0}, remote_dram_hits_{0};
    std::atomic<size_t> misses_{0};
};

}  // namespace distributed
}  // namespace cascade

/**
 * Cascade Distributed Backend Implementation
 * 
 * Multi-Node Multi-GPU KV Cache with GPU-aware MPI
 * Optimized for HPE Slingshot on Perlmutter
 */

#include "cascade_distributed.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <iostream>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
    } \
} while(0)

namespace cascade {
namespace distributed {

// ============================================================================
// DistributedDRAMBackend Implementation
// ============================================================================

#ifdef USE_MPI
DistributedDRAMBackend::DistributedDRAMBackend(size_t capacity, MPI_Comm comm)
    : capacity_(capacity), comm_(comm) {
    
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &world_size_);
    
    // Allocate pinned memory for RMA window (GPU-aware MPI requires pinned)
    CUDA_CHECK(cudaHostAlloc(&dram_base_, capacity_, cudaHostAllocDefault));
    memset(dram_base_, 0, capacity_);
    
    // Create MPI window for one-sided communication
    MPI_Win_create(dram_base_, capacity_, 1, MPI_INFO_NULL, comm_, &window_);
    
    if (rank_ == 0) {
        printf("[DRAM Backend] Initialized %.2f GB per node, %d nodes total\n",
               capacity_ / (1024.0 * 1024.0 * 1024.0), world_size_);
    }
}
#else
DistributedDRAMBackend::DistributedDRAMBackend(size_t capacity)
    : capacity_(capacity) {
    CUDA_CHECK(cudaHostAlloc(&dram_base_, capacity_, cudaHostAllocDefault));
    memset(dram_base_, 0, capacity_);
}
#endif

DistributedDRAMBackend::~DistributedDRAMBackend() {
#ifdef USE_MPI
    MPI_Barrier(comm_);
    MPI_Win_free(&window_);
#endif
    if (dram_base_) {
        cudaFreeHost(dram_base_);
    }
}

bool DistributedDRAMBackend::put_local(const BlockId& id, const uint8_t* data, size_t size) {
    size_t offset = write_offset_.fetch_add(size);
    if (offset + size > capacity_) {
        return false;  // Out of space
    }
    
    memcpy(static_cast<uint8_t*>(dram_base_) + offset, data, size);
    
    DRAMBlock block{offset, size};
    index_.put(id, block, size);
    used_.fetch_add(size);
    
    return true;
}

bool DistributedDRAMBackend::get_local(const BlockId& id, uint8_t* out, size_t* out_size) {
    auto block = index_.get(id);
    if (!block) return false;
    
    memcpy(out, static_cast<uint8_t*>(dram_base_) + block->offset, block->size);
    *out_size = block->size;
    return true;
}

bool DistributedDRAMBackend::get_remote(int target_rank, size_t offset, 
                                         uint8_t* out, size_t size) {
#ifdef USE_MPI
    MPI_Win_lock(MPI_LOCK_SHARED, target_rank, 0, window_);
    MPI_Get(out, size, MPI_BYTE, target_rank, offset, size, MPI_BYTE, window_);
    MPI_Win_unlock(target_rank, window_);
    return true;
#else
    return false;
#endif
}

void DistributedDRAMBackend::barrier() {
#ifdef USE_MPI
    MPI_Barrier(comm_);
#endif
}

// ============================================================================
// DistributedGPUBackend Implementation
// ============================================================================

#ifdef USE_MPI
DistributedGPUBackend::DistributedGPUBackend(size_t cap_per_gpu, int num_gpus, MPI_Comm comm)
    : cap_per_gpu_(cap_per_gpu), num_gpus_(num_gpus), comm_(comm) {
    
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &world_size_);
#else
DistributedGPUBackend::DistributedGPUBackend(size_t cap_per_gpu, int num_gpus)
    : cap_per_gpu_(cap_per_gpu), num_gpus_(num_gpus) {
#endif
    
    // Initialize local GPU backends
    for (int i = 0; i < num_gpus_; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        gpus_.push_back(std::make_unique<GPUBackend>(cap_per_gpu_, i));
    }
    
    // Setup NVLink peer access
    setup_nvlink();
    
    // Allocate pinned staging buffers
    CUDA_CHECK(cudaSetDevice(0));
    for (int i = 0; i < 32; i++) {
        CUDA_CHECK(cudaHostAlloc(&pinned_[i], staging_size_, cudaHostAllocDefault));
    }
    
    // Initialize MPI RMA window
    init_window();
    
    if (rank_ == 0) {
        printf("[GPU Backend] %d GPUs/node, %.2f GB/GPU, %d nodes = %.2f TB total\n",
               num_gpus_, cap_per_gpu_ / (1024.0 * 1024.0 * 1024.0), world_size_,
               (num_gpus_ * cap_per_gpu_ * world_size_) / (1024.0 * 1024.0 * 1024.0 * 1024.0));
    }
}

DistributedGPUBackend::~DistributedGPUBackend() {
#ifdef USE_MPI
    MPI_Barrier(comm_);
    if (window_ != MPI_WIN_NULL) {
        MPI_Win_free(&window_);
    }
#endif
    for (int i = 0; i < 32; i++) {
        if (pinned_[i]) cudaFreeHost(pinned_[i]);
    }
}

void DistributedGPUBackend::setup_nvlink() {
    // Check and enable peer access between GPUs
    for (int i = 0; i < num_gpus_; i++) {
        for (int j = 0; j < num_gpus_; j++) {
            if (i != j) {
                int can_access;
                CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, i, j));
                peer_[i][j] = (can_access != 0);
                if (can_access) {
                    CUDA_CHECK(cudaSetDevice(i));
                    cudaDeviceEnablePeerAccess(j, 0);  // May fail if already enabled
                }
            }
        }
    }
    
    if (rank_ == 0) {
        printf("[NVLink] Peer access matrix:\n");
        for (int i = 0; i < num_gpus_; i++) {
            printf("  GPU %d: ", i);
            for (int j = 0; j < num_gpus_; j++) {
                printf("%c ", peer_[i][j] ? 'Y' : 'N');
            }
            printf("\n");
        }
    }
}

void DistributedGPUBackend::init_window() {
#ifdef USE_MPI
    // Use first pinned buffer as RMA window
    MPI_Win_create(pinned_[0], staging_size_, 1, MPI_INFO_NULL, comm_, &window_);
#endif
}

int DistributedGPUBackend::get_target_node(const BlockId& id) const {
    // Consistent hash: first 2 bytes → node
    if (id.size() < 2) return 0;
    uint16_t h = (static_cast<uint16_t>(static_cast<uint8_t>(id[0])) << 8) 
               | static_cast<uint8_t>(id[1]);
    return h % world_size_;
}

int DistributedGPUBackend::get_target_gpu(const BlockId& id) const {
    // Bytes 2-3 → GPU within node
    if (id.size() < 4) return 0;
    uint16_t h = (static_cast<uint16_t>(static_cast<uint8_t>(id[2])) << 8) 
               | static_cast<uint8_t>(id[3]);
    return h % num_gpus_;
}

bool DistributedGPUBackend::put(const BlockId& id, const uint8_t* data, size_t size) {
    int target_node = get_target_node(id);
    int target_gpu = get_target_gpu(id);
    
    if (target_node == rank_) {
        // Local put
        return put_local(id, data, size, target_gpu);
    } else {
        // Remote put via MPI RMA
#ifdef USE_MPI
        // Copy data to staging buffer
        size_t chunk = std::min(size, staging_size_);
        memcpy(pinned_[0], data, chunk);
        
        // Send to remote node's staging buffer
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, target_node, 0, window_);
        MPI_Put(pinned_[0], chunk, MPI_BYTE, target_node, 0, chunk, MPI_BYTE, window_);
        MPI_Win_unlock(target_node, window_);
        
        // TODO: Need a protocol for remote node to store from staging to GPU
        return true;
#else
        return false;
#endif
    }
}

bool DistributedGPUBackend::put_local(const BlockId& id, const uint8_t* data, 
                                       size_t size, int gpu) {
    if (gpu < 0 || gpu >= num_gpus_) return false;
    
    CUDA_CHECK(cudaSetDevice(gpu));
    bool ok = gpus_[gpu]->put(id, data, size);
    
    if (ok) {
        // Update global index
        BlockLocation loc;
        loc.node_id = rank_;
        loc.gpu_id = gpu;
        loc.offset = 0;  // TODO: track actual offset
        loc.size = size;
        loc.timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
        loc.is_gpu = true;
        global_index_.put(id, loc, size);
    }
    
    return ok;
}

bool DistributedGPUBackend::get(const BlockId& id, uint8_t* out, size_t* out_size) {
    // First try local
    if (get_local(id, out, out_size)) {
        return true;
    }
    
    // Check global index for remote location
    auto loc = global_index_.get(id);
    if (!loc) return false;
    
    if (loc->node_id == rank_) {
        // Should have found it locally - inconsistent state
        return false;
    }
    
    // Remote get
    return get_remote(loc->node_id, loc->offset, loc->size, out);
}

bool DistributedGPUBackend::get_local(const BlockId& id, uint8_t* out, size_t* out_size) {
    // Check all local GPUs
    for (int i = 0; i < num_gpus_; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        if (gpus_[i]->get(id, out, out_size)) {
            return true;
        }
    }
    return false;
}

bool DistributedGPUBackend::get_remote(int target_rank, size_t offset, 
                                        size_t size, uint8_t* out) {
#ifdef USE_MPI
    // RMA Get to staging buffer
    MPI_Win_lock(MPI_LOCK_SHARED, target_rank, 0, window_);
    MPI_Get(pinned_[1], size, MPI_BYTE, target_rank, offset, size, MPI_BYTE, window_);
    MPI_Win_unlock(target_rank, window_);
    
    // Copy from staging to output (could be GPU memory)
    CUDA_CHECK(cudaMemcpy(out, pinned_[1], size, cudaMemcpyDefault));
    return true;
#else
    return false;
#endif
}

std::optional<BlockLocation> DistributedGPUBackend::locate(const BlockId& id) const {
    return global_index_.get(id);
}

size_t DistributedGPUBackend::used_bytes() const {
    size_t total = 0;
    for (const auto& gpu : gpus_) {
        total += gpu->used_bytes();
    }
    return total;
}

void DistributedGPUBackend::sync_all() {
    for (int i = 0; i < num_gpus_; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        gpus_[i]->sync_all();
    }
}

void DistributedGPUBackend::barrier() {
#ifdef USE_MPI
    MPI_Barrier(comm_);
#endif
}

// ============================================================================
// DistributedStore Implementation
// ============================================================================

#ifdef USE_MPI
DistributedStore::DistributedStore(const DistributedConfig& cfg, MPI_Comm comm)
    : cfg_(cfg), comm_(comm) {
    
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &world_size_);
#else
DistributedStore::DistributedStore(const DistributedConfig& cfg)
    : cfg_(cfg), rank_(0), world_size_(1) {
#endif
    
#ifdef USE_MPI
    gpu_ = std::make_unique<DistributedGPUBackend>(
        cfg_.gpu_capacity_per_device, cfg_.num_gpus_per_node, comm_);
    dram_ = std::make_unique<DistributedDRAMBackend>(cfg_.dram_capacity, comm_);
#else
    gpu_ = std::make_unique<DistributedGPUBackend>(
        cfg_.gpu_capacity_per_device, cfg_.num_gpus_per_node);
    dram_ = std::make_unique<DistributedDRAMBackend>(cfg_.dram_capacity);
#endif
    
    if (rank_ == 0) {
        printf("=== Cascade Distributed Store Initialized ===\n");
        printf("  Nodes: %d\n", world_size_);
        printf("  GPUs/node: %d\n", cfg_.num_gpus_per_node);
        printf("  GPU capacity: %.2f GB/device\n", 
               cfg_.gpu_capacity_per_device / (1024.0 * 1024.0 * 1024.0));
        printf("  DRAM capacity: %.2f GB/node\n",
               cfg_.dram_capacity / (1024.0 * 1024.0 * 1024.0));
        printf("  Total cluster GPU: %.2f TB\n",
               (cfg_.gpu_capacity_per_device * cfg_.num_gpus_per_node * world_size_) 
               / (1024.0 * 1024.0 * 1024.0 * 1024.0));
        printf("=============================================\n");
    }
}

DistributedStore::~DistributedStore() {
    barrier();
}

bool DistributedStore::put(const BlockId& id, const uint8_t* data, size_t size) {
    // Try GPU first
    if (gpu_->put(id, data, size)) {
        return true;
    }
    
    // Fall back to DRAM
    return dram_->put_local(id, data, size);
}

bool DistributedStore::get(const BlockId& id, uint8_t* out, size_t* out_size) {
    // Try local GPU
    if (gpu_->get_local(id, out, out_size)) {
        local_gpu_hits_++;
        return true;
    }
    
    // Try local DRAM
    if (dram_->get_local(id, out, out_size)) {
        local_dram_hits_++;
        return true;
    }
    
    // Try remote
    if (gpu_->get(id, out, out_size)) {
        remote_gpu_hits_++;
        return true;
    }
    
    misses_++;
    return false;
}

bool DistributedStore::contains(const BlockId& id) const {
    return gpu_->locate(id).has_value();
}

std::optional<BlockLocation> DistributedStore::locate(const BlockId& id) const {
    return gpu_->locate(id);
}

size_t DistributedStore::put_batch(const std::vector<BlockId>& ids,
                                    const std::vector<const uint8_t*>& data,
                                    const std::vector<size_t>& sizes) {
    size_t success = 0;
    for (size_t i = 0; i < ids.size(); i++) {
        if (put(ids[i], data[i], sizes[i])) {
            success++;
        }
    }
    return success;
}

size_t DistributedStore::get_batch(const std::vector<BlockId>& ids,
                                    std::vector<uint8_t*>& out,
                                    std::vector<size_t>& sizes) {
    size_t success = 0;
    for (size_t i = 0; i < ids.size(); i++) {
        if (get(ids[i], out[i], &sizes[i])) {
            success++;
        }
    }
    return success;
}

void DistributedStore::barrier() {
#ifdef USE_MPI
    MPI_Barrier(comm_);
#endif
}

void DistributedStore::sync_metadata() {
    // TODO: Implement global metadata sync
    barrier();
}

DistributedStore::Stats DistributedStore::get_stats() {
    Stats s;
    s.local_gpu_used = gpu_->used_bytes();
    s.local_dram_used = dram_->used_bytes();
    s.local_gpu_hits = local_gpu_hits_.load();
    s.local_dram_hits = local_dram_hits_.load();
    s.remote_gpu_hits = remote_gpu_hits_.load();
    s.remote_dram_hits = remote_dram_hits_.load();
    s.misses = misses_.load();
    
#ifdef USE_MPI
    // Aggregate across cluster
    MPI_Allreduce(&s.local_gpu_used, &s.cluster_gpu_used, 1, 
                  MPI_UNSIGNED_LONG, MPI_SUM, comm_);
    MPI_Allreduce(&s.local_dram_used, &s.cluster_dram_used, 1,
                  MPI_UNSIGNED_LONG, MPI_SUM, comm_);
#else
    s.cluster_gpu_used = s.local_gpu_used;
    s.cluster_dram_used = s.local_dram_used;
#endif
    
    return s;
}

}  // namespace distributed
}  // namespace cascade

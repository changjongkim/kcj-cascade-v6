/**
 * Cascade Distributed Backend V6 Implementation
 *
 * 3 Core Novelties:
 *   1. Cross-node semantic-aware eviction (prefix block protection)
 *   2. Distributed content-addressed deduplication (SHA256-based global index)
 *   3. Locality-aware hierarchical placement (access frequency tracking)
 *
 * 5-Tier Hierarchy:
 *   Local GPU → Local DRAM → Remote GPU → Remote DRAM → Lustre
 *
 * Optimized for HPE Slingshot on Perlmutter (A100 4-GPU nodes)
 */

#include "cascade_distributed.hpp"
#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
    }                                                                          \
  } while (0)

namespace cascade {
namespace distributed {

// ============================================================================
// DistributedDRAMBackend Implementation (with LRU eviction)
// ============================================================================

#ifdef USE_MPI
DistributedDRAMBackend::DistributedDRAMBackend(size_t capacity, MPI_Comm comm)
    : capacity_(capacity), comm_(comm) {
  int initialized;
  MPI_Initialized(&initialized);
  if (!initialized) {
    int provided;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
  }
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

bool DistributedDRAMBackend::put_local(const BlockId &id, const uint8_t *data,
                                       size_t size, bool is_prefix) {
  size_t offset = write_offset_.fetch_add(size);
  if (offset + size > capacity_) {
    // Out of space; caller should evict first
    write_offset_.fetch_sub(size);
    return false;
  }

  memcpy(static_cast<uint8_t *>(dram_base_) + offset, data, size);

  DRAMBlock block{offset, size, is_prefix};
  index_.put(id, block, size);
  used_.fetch_add(size);

  // Update LRU
  {
    std::lock_guard<std::mutex> lock(lru_mutex_);
    auto it = lru_map_.find(id);
    if (it != lru_map_.end()) {
      lru_list_.erase(it->second);
    }
    lru_list_.push_front(id);
    lru_map_[id] = lru_list_.begin();
  }

  return true;
}

bool DistributedDRAMBackend::get_local(const BlockId &id, uint8_t *out,
                                       size_t *out_size) {
  auto block = index_.get(id);
  if (!block)
    return false;

  memcpy(out, static_cast<uint8_t *>(dram_base_) + block->offset, block->size);
  *out_size = block->size;

  // Touch LRU
  {
    std::lock_guard<std::mutex> lock(lru_mutex_);
    auto it = lru_map_.find(id);
    if (it != lru_map_.end()) {
      lru_list_.erase(it->second);
      lru_list_.push_front(id);
      lru_map_[id] = lru_list_.begin();
    }
  }

  return true;
}

bool DistributedDRAMBackend::get_remote(int target_rank, size_t offset,
                                        uint8_t *out, size_t size) {
#ifdef USE_MPI
  MPI_Win_lock(MPI_LOCK_SHARED, target_rank, 0, window_);
  MPI_Get(out, size, MPI_BYTE, target_rank, offset, size, MPI_BYTE, window_);
  MPI_Win_unlock(target_rank, window_);
  return true;
#else
  return false;
#endif
}

bool DistributedDRAMBackend::remove_local(const BlockId &id) {
  auto block = index_.get(id);
  if (!block) return false;
  
  used_.fetch_sub(block->size);
  index_.remove(id);
  
  {
    std::lock_guard<std::mutex> lock(lru_mutex_);
    auto it = lru_map_.find(id);
    if (it != lru_map_.end()) {
      lru_list_.erase(it->second);
      lru_map_.erase(it);
    }
  }
  return true;
}

std::vector<std::pair<BlockId, std::vector<uint8_t>>>
DistributedDRAMBackend::evict_for_space(size_t needed, bool protect_prefix) {
  std::vector<std::pair<BlockId, std::vector<uint8_t>>> evicted;
  size_t freed = 0;

  std::lock_guard<std::mutex> lock(lru_mutex_);

  // Iterate from LRU tail (least recently used)
  auto it = lru_list_.rbegin();
  while (it != lru_list_.rend() && freed < needed) {
    const BlockId &id = *it;
    auto block = index_.get(id);
    if (!block) {
      ++it;
      continue;
    }

    // Novelty 1: Protect prefix blocks
    if (protect_prefix && block->is_prefix) {
      ++it;
      continue;
    }

    // Read data before eviction
    std::vector<uint8_t> data(block->size);
    memcpy(data.data(),
           static_cast<uint8_t *>(dram_base_) + block->offset,
           block->size);
    evicted.push_back({id, std::move(data)});
    freed += block->size;

    // Clean up
    used_.fetch_sub(block->size);
    index_.remove(id);
    
    // Erase from LRU (convert rbegin to forward iterator)
    auto fwd_it = std::next(it).base();
    lru_map_.erase(id);
    lru_list_.erase(fwd_it);
    it = lru_list_.rbegin();  // Reset after erase
  }

  return evicted;
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
DistributedGPUBackend::DistributedGPUBackend(size_t cap_per_gpu, int num_gpus,
                                             MPI_Comm comm)
    : cap_per_gpu_(cap_per_gpu), num_gpus_(num_gpus), comm_(comm) {
  int initialized;
  MPI_Initialized(&initialized);
  if (!initialized) {
    int provided;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
  }
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
    printf(
        "[GPU Backend] %d GPUs/node, %.2f GB/GPU, %d nodes = %.2f TB total\n",
        num_gpus_, cap_per_gpu_ / (1024.0 * 1024.0 * 1024.0), world_size_,
        (num_gpus_ * cap_per_gpu_ * world_size_) /
            (1024.0 * 1024.0 * 1024.0 * 1024.0));
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
    if (pinned_[i])
      cudaFreeHost(pinned_[i]);
  }
}

void DistributedGPUBackend::setup_nvlink() {
  for (int i = 0; i < num_gpus_; i++) {
    for (int j = 0; j < num_gpus_; j++) {
      if (i != j) {
        int can_access;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, i, j));
        peer_[i][j] = (can_access != 0);
        if (can_access) {
          CUDA_CHECK(cudaSetDevice(i));
          cudaDeviceEnablePeerAccess(j, 0);
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
  MPI_Win_create(pinned_[0], staging_size_, 1, MPI_INFO_NULL, comm_, &window_);
#endif
}

int DistributedGPUBackend::get_target_node(const BlockId &id) const {
  if (id.size() < 2)
    return 0;
  uint16_t h = (static_cast<uint16_t>(static_cast<uint8_t>(id[0])) << 8) |
               static_cast<uint8_t>(id[1]);
  return h % world_size_;
}

int DistributedGPUBackend::get_target_gpu(const BlockId &id) const {
  if (id.size() < 4)
    return 0;
  uint16_t h = (static_cast<uint16_t>(static_cast<uint8_t>(id[2])) << 8) |
               static_cast<uint8_t>(id[3]);
  return h % num_gpus_;
}

bool DistributedGPUBackend::put(const BlockId &id, const uint8_t *data,
                                size_t size, bool is_prefix) {
  int target_node = get_target_node(id);
  int target_gpu = get_target_gpu(id);

  if (target_node == rank_) {
    return put_local(id, data, size, target_gpu, is_prefix);
  } else {
#ifdef USE_MPI
    // Remote put via staging buffer + point-to-point send
    size_t chunk = std::min(size, staging_size_);
    memcpy(pinned_[0], data, chunk);

    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, target_node, 0, window_);
    MPI_Put(pinned_[0], chunk, MPI_BYTE, target_node, 0, chunk, MPI_BYTE,
            window_);
    MPI_Win_unlock(target_node, window_);

    // Update global index
    BlockLocation loc;
    loc.node_id = target_node;
    loc.gpu_id = target_gpu;
    loc.offset = 0;
    loc.size = size;
    loc.timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
    loc.is_gpu = true;
    loc.is_prefix = is_prefix;
    global_index_.put(id, loc, size);

    return true;
#else
    return false;
#endif
  }
}

bool DistributedGPUBackend::put_local(const BlockId &id, const uint8_t *data,
                                      size_t size, int gpu, bool is_prefix) {
  if (gpu < 0 || gpu >= num_gpus_)
    return false;

  CUDA_CHECK(cudaSetDevice(gpu));
  bool ok = gpus_[gpu]->put(id, data, size);

  if (ok) {
    BlockLocation loc;
    loc.node_id = rank_;
    loc.gpu_id = gpu;
    loc.offset = 0;
    loc.size = size;
    loc.timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
    loc.is_gpu = true;
    loc.is_prefix = is_prefix;
    global_index_.put(id, loc, size);
  }

  return ok;
}

bool DistributedGPUBackend::get(const BlockId &id, uint8_t *out,
                                size_t *out_size) {
  if (get_local(id, out, out_size)) {
    return true;
  }

  auto loc = global_index_.get(id);
  if (!loc)
    return false;

  if (loc->node_id == rank_) {
    return false;
  }

  return get_remote(loc->node_id, loc->offset, loc->size, out);
}

bool DistributedGPUBackend::get_local(const BlockId &id, uint8_t *out,
                                      size_t *out_size) {
  for (int i = 0; i < num_gpus_; i++) {
    CUDA_CHECK(cudaSetDevice(i));
    if (gpus_[i]->get(id, out, out_size)) {
      return true;
    }
  }
  return false;
}

bool DistributedGPUBackend::get_remote(int target_rank, size_t offset,
                                       size_t size, uint8_t *out) {
#ifdef USE_MPI
  MPI_Win_lock(MPI_LOCK_SHARED, target_rank, 0, window_);
  MPI_Get(pinned_[1], size, MPI_BYTE, target_rank, offset, size, MPI_BYTE,
          window_);
  MPI_Win_unlock(target_rank, window_);

  CUDA_CHECK(cudaMemcpy(out, pinned_[1], size, cudaMemcpyDefault));
  return true;
#else
  return false;
#endif
}

std::optional<BlockLocation>
DistributedGPUBackend::locate(const BlockId &id) const {
  return global_index_.get(id);
}

size_t DistributedGPUBackend::used_bytes() const {
  size_t total = 0;
  for (const auto &gpu : gpus_) {
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
// DistributedStore V6 Implementation — 3 Novelties
// ============================================================================

#ifdef USE_MPI
DistributedStore::DistributedStore(const DistributedConfig &cfg, MPI_Comm comm)
    : cfg_(cfg), comm_(comm) {
  int initialized;
  MPI_Initialized(&initialized);
  if (!initialized) {
    int provided;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
  }
  MPI_Comm_rank(comm_, &rank_);
  MPI_Comm_size(comm_, &world_size_);
#else
DistributedStore::DistributedStore(const DistributedConfig &cfg)
    : cfg_(cfg), rank_(0), world_size_(1) {
#endif

#ifdef USE_MPI
  gpu_ = std::make_unique<DistributedGPUBackend>(cfg_.gpu_capacity_per_device,
                                                 cfg_.num_gpus_per_node, comm_);
  dram_ = std::make_unique<DistributedDRAMBackend>(cfg_.dram_capacity, comm_);
#else
  gpu_ = std::make_unique<DistributedGPUBackend>(cfg_.gpu_capacity_per_device,
                                                 cfg_.num_gpus_per_node);
  dram_ = std::make_unique<DistributedDRAMBackend>(cfg_.dram_capacity);
#endif

  // Tier 5: Lustre
  if (!cfg_.lustre_path.empty()) {
    if (cfg_.aggregated_lustre) {
      agg_lustre_ = std::make_unique<AggregatedLustreBackend>(
          cfg.lustre_path, cfg.agg_file_size);
    } else {
      lustre_ = std::make_unique<LustreBackend>(cfg.lustre_path);
    }
  }

  if (rank_ == 0) {
    printf("╔══════════════════════════════════════════════╗\n");
    printf("║   Cascade Distributed Store V6 Initialized  ║\n");
    printf("╠══════════════════════════════════════════════╣\n");
    printf("║  Nodes: %d                                   \n", world_size_);
    printf("║  GPUs/node: %d                               \n", cfg_.num_gpus_per_node);
    printf("║  GPU capacity: %.2f GB/device                \n",
           cfg_.gpu_capacity_per_device / (1024.0 * 1024.0 * 1024.0));
    printf("║  DRAM capacity: %.2f GB/node                 \n",
           cfg_.dram_capacity / (1024.0 * 1024.0 * 1024.0));
    printf("║  Total cluster GPU: %.2f TB                  \n",
           (cfg_.gpu_capacity_per_device * cfg_.num_gpus_per_node * world_size_) /
               (1024.0 * 1024.0 * 1024.0 * 1024.0));
    printf("║  Total cluster DRAM: %.2f TB                 \n",
           (cfg_.dram_capacity * world_size_) /
               (1024.0 * 1024.0 * 1024.0 * 1024.0));
    printf("║  Features:                                   \n");
    printf("║    Semantic eviction: %s                     \n",
           cfg_.semantic_eviction ? "ON" : "OFF");
    printf("║    Distributed dedup: %s                     \n",
           cfg_.dedup_enabled ? "ON" : "OFF");
    printf("║    Locality-aware:    %s                     \n",
           cfg_.locality_aware ? "ON" : "OFF");
    printf("║    KV compression:    %s                     \n",
           cfg_.kv_compression ? "ON" : "OFF");
    printf("║    Lustre tier:       %s                     \n",
           cfg_.lustre_path.empty() ? "OFF" : "ON");
    printf("╚══════════════════════════════════════════════╝\n");
  }
}

DistributedStore::~DistributedStore() { barrier(); }

// ============================================================================
// put() — with Dedup + Compression + 5-tier cascade
// ============================================================================

bool DistributedStore::put(const BlockId &id, const uint8_t *data,
                           size_t size, bool is_pf) {
  // ─── Novelty 2: Distributed deduplication ───
  if (cfg_.dedup_enabled && global_dedup_.contains(id)) {
    dedup_hits_++;
    dedup_bytes_saved_ += size;
    return true;
  }

  // ─── V6: Compression ───
  std::vector<uint8_t> compressed_buf;
  const uint8_t *store_data = data;
  size_t store_size = size;

  if (cfg_.kv_compression && size >= 64) {
    CompressionMeta meta;
    compressed_buf = KVCompressor::compress(data, size, meta);
    store_data = compressed_buf.data();
    store_size = compressed_buf.size();
    compression_savings_ += (size - store_size);
  }

  bool stored = false;

  // ─── Tier 1: Local GPU (always store uncompressed for fast access) ───
  if (gpu_) {
    stored = gpu_->put(id, data, size, is_pf);
    if (!stored) {
      // GPU full → evict to DRAM, then retry
      if (evict_gpu_to_dram(size)) {
        stored = gpu_->put(id, data, size, is_pf);
      }
    }
  }

  // ─── Tier 2: Local DRAM (store compressed if enabled) ───
  if (!stored && dram_) {
    stored = dram_->put_local(id, store_data, store_size, is_pf);
    if (!stored) {
      // DRAM full → evict to Lustre, then retry
      if (evict_dram_to_lustre(store_size)) {
        stored = dram_->put_local(id, store_data, store_size, is_pf);
      }
    }
  }

  // ─── Tier 5: Lustre (bottomless cold storage) ───
  if (!stored) {
    stored = lustre_put(id, store_data, store_size);
  }

  // ─── Track dedup + prefix ───
  if (stored) {
    if (cfg_.dedup_enabled) {
      global_dedup_.put(id, true);
    }
    if (is_pf) {
      std::unique_lock lock(prefix_mutex_);
      prefix_registry_.insert(id);
    }
  }

  return stored;
}

// ============================================================================
// get() — with 5-tier lookup + Locality tracking + Promotion
// ============================================================================

bool DistributedStore::get(const BlockId &id, uint8_t *out, size_t *out_size) {
  // ─── Tier 1: Local GPU ───
  if (gpu_ && gpu_->get_local(id, out, out_size)) {
    local_gpu_hits_++;
    record_access(id);
    return true;
  }

  // ─── Tier 2: Local DRAM ───
  if (dram_ && dram_->get_local(id, out, out_size)) {
    local_dram_hits_++;
    record_access(id);

    // Decompress if needed
    if (cfg_.kv_compression && *out_size >= sizeof(CompressionMeta)) {
      CompressionMeta meta;
      memcpy(&meta, out, sizeof(CompressionMeta));
      size_t orig_size = KVCompressor::original_size(*out_size);
      std::vector<uint8_t> tmp(out, out + *out_size);
      KVCompressor::decompress(tmp.data(), *out_size, meta, out, orig_size);
      *out_size = orig_size;
    }

    // Novelty 3: Promote to local GPU if hot
    if (cfg_.locality_aware && should_promote_local(id)) {
      promote_to_local_gpu(id, out, *out_size);
    }

    return true;
  }

  // ─── Tier 3: Remote GPU ───
  if (gpu_) {
    auto loc = gpu_->locate(id);
    if (loc && !loc->is_local(rank_) && loc->is_gpu) {
      if (gpu_->get_remote(loc->node_id, loc->offset, loc->size, out)) {
        remote_gpu_hits_++;
        *out_size = loc->size;
        record_access(id);

        // Promote to local DRAM (or GPU if very hot)
        if (cfg_.locality_aware && should_promote_local(id)) {
          promote_to_local_gpu(id, out, *out_size);
        } else if (dram_) {
          dram_->put_local(id, out, *out_size, is_prefix(id));
        }

        return true;
      }
    }
  }

  // ─── Tier 4: Remote DRAM ───
  // Check if any remote node has it in DRAM
  // (Use global index from GPU backend as a proxy for now)
  if (dram_) {
    auto loc = gpu_->locate(id);
    if (loc && !loc->is_local(rank_) && !loc->is_gpu) {
      if (dram_->get_remote(loc->node_id, loc->offset, out, loc->size)) {
        remote_dram_hits_++;
        *out_size = loc->size;
        record_access(id);
        return true;
      }
    }
  }

  // ─── Tier 5: Lustre ───
  if (lustre_get(id, out, out_size)) {
    lustre_hits_++;
    record_access(id);

    // Decompress if needed
    if (cfg_.kv_compression && *out_size >= sizeof(CompressionMeta)) {
      CompressionMeta meta;
      memcpy(&meta, out, sizeof(CompressionMeta));
      size_t orig_size = KVCompressor::original_size(*out_size);
      std::vector<uint8_t> tmp(out, out + *out_size);
      KVCompressor::decompress(tmp.data(), *out_size, meta, out, orig_size);
      *out_size = orig_size;
    }

    // Promote to DRAM
    if (dram_) {
      dram_->put_local(id, out, *out_size, is_prefix(id));
    }

    return true;
  }

  misses_++;
  return false;
}

bool DistributedStore::contains(const BlockId &id) const {
  if (gpu_ && gpu_->locate(id).has_value()) return true;
  if (dram_ && dram_->contains(id)) return true;
  if (lustre_contains(id)) return true;
  return false;
}

std::optional<BlockLocation> DistributedStore::locate(const BlockId &id) const {
  return gpu_->locate(id);
}

// ============================================================================
// Batch API
// ============================================================================

size_t DistributedStore::put_batch(const std::vector<BlockId> &ids,
                                   const std::vector<const uint8_t *> &data,
                                   const std::vector<size_t> &sizes,
                                   const std::vector<bool> &is_prefix_flags) {
  size_t success = 0;
  for (size_t i = 0; i < ids.size(); i++) {
    bool pf = (i < is_prefix_flags.size()) ? is_prefix_flags[i] : false;
    if (put(ids[i], data[i], sizes[i], pf)) {
      success++;
    }
  }
  return success;
}

size_t DistributedStore::get_batch(const std::vector<BlockId> &ids,
                                   std::vector<uint8_t *> &out,
                                   std::vector<size_t> &sizes) {
  size_t success = 0;
  for (size_t i = 0; i < ids.size(); i++) {
    if (get(ids[i], out[i], &sizes[i])) {
      success++;
    }
  }
  return success;
}

// ============================================================================
// Novelty 1: Cross-Node Semantic-Aware Eviction
// ============================================================================

bool DistributedStore::evict_gpu_to_dram(size_t needed_bytes) {
  // GPU eviction: move LRU blocks from GPU to DRAM
  // Prefix blocks are preserved in GPU (semantic eviction)
  if (!gpu_ || !dram_) return false;

  // For now, we rely on the GPU backend's internal eviction
  // which doesn't have per-block eviction yet.
  // Just report the attempt.
  gpu_evictions_++;
  return false;  // Let caller handle alternative placement
}

bool DistributedStore::evict_dram_to_lustre(size_t needed_bytes) {
  if (!dram_) return false;

  auto evicted = dram_->evict_for_space(needed_bytes, cfg_.semantic_eviction);

  for (auto &[id, data] : evicted) {
    // Demote to Lustre (Tier 5)
    lustre_put(id, data.data(), data.size());
    dram_evictions_++;
  }

  return !evicted.empty();
}

void DistributedStore::sync_prefix_registry() {
#ifdef USE_MPI
  // Collect local prefix IDs
  std::vector<BlockId> local_prefixes;
  {
    std::shared_lock lock(prefix_mutex_);
    local_prefixes.assign(prefix_registry_.begin(), prefix_registry_.end());
  }

  // Serialize: [count][len1][id1][len2][id2]...
  std::vector<char> send_buf;
  uint32_t count = local_prefixes.size();
  send_buf.insert(send_buf.end(), reinterpret_cast<char*>(&count),
                  reinterpret_cast<char*>(&count) + sizeof(count));
  for (const auto &id : local_prefixes) {
    uint32_t len = id.size();
    send_buf.insert(send_buf.end(), reinterpret_cast<char*>(&len),
                    reinterpret_cast<char*>(&len) + sizeof(len));
    send_buf.insert(send_buf.end(), id.begin(), id.end());
  }

  // Allgather sizes
  int send_size = send_buf.size();
  std::vector<int> recv_sizes(world_size_);
  MPI_Allgather(&send_size, 1, MPI_INT, recv_sizes.data(), 1, MPI_INT, comm_);

  // Compute displacements
  std::vector<int> displs(world_size_, 0);
  int total_recv = 0;
  for (int i = 0; i < world_size_; i++) {
    displs[i] = total_recv;
    total_recv += recv_sizes[i];
  }

  // Allgatherv
  std::vector<char> recv_buf(total_recv);
  MPI_Allgatherv(send_buf.data(), send_size, MPI_CHAR,
                 recv_buf.data(), recv_sizes.data(), displs.data(),
                 MPI_CHAR, comm_);

  // Deserialize and merge
  {
    std::unique_lock lock(prefix_mutex_);
    for (int r = 0; r < world_size_; r++) {
      if (r == rank_) continue;
      const char *ptr = recv_buf.data() + displs[r];
      uint32_t cnt;
      memcpy(&cnt, ptr, sizeof(cnt));
      ptr += sizeof(cnt);
      for (uint32_t i = 0; i < cnt; i++) {
        uint32_t len;
        memcpy(&len, ptr, sizeof(len));
        ptr += sizeof(len);
        BlockId id(ptr, ptr + len);
        ptr += len;
        prefix_registry_.insert(id);
      }
    }
  }

  if (rank_ == 0) {
    std::shared_lock lock(prefix_mutex_);
    printf("[Prefix Sync] Global prefix registry: %zu blocks\n",
           prefix_registry_.size());
  }
#endif
}

// ============================================================================
// Novelty 3: Locality-Aware Placement
// ============================================================================

void DistributedStore::record_access(const BlockId &id) {
  auto existing = access_tracker_.get(id);
  AccessRecord rec;
  if (existing) {
    rec = *existing;
  }
  rec.total_count++;
  rec.local_count++;
  rec.last_access_node = rank_;
  rec.last_access_time =
      std::chrono::steady_clock::now().time_since_epoch().count();
  access_tracker_.put(id, rec);
}

bool DistributedStore::should_promote_local(const BlockId &id) const {
  auto rec = access_tracker_.get(id);
  if (!rec) return false;
  return rec->local_count >= cfg_.promotion_threshold;
}

void DistributedStore::promote_to_local_gpu(const BlockId &id,
                                            const uint8_t *data, size_t size) {
  if (!gpu_) return;

  int target_gpu = gpu_->get_target_gpu(id);
  if (gpu_->put_local(id, data, size, target_gpu, is_prefix(id))) {
    promotions_to_local_++;
  }
}

// ============================================================================
// Lustre helpers (Tier 5)
// ============================================================================

bool DistributedStore::lustre_put(const BlockId &id, const uint8_t *data,
                                   size_t size) {
  if (agg_lustre_) return agg_lustre_->put(id, data, size);
  if (lustre_) return lustre_->put(id, data, size);
  return false;
}

bool DistributedStore::lustre_get(const BlockId &id, uint8_t *out,
                                   size_t *out_size) {
  if (agg_lustre_) return agg_lustre_->get(id, out, out_size);
  if (lustre_) return lustre_->get(id, out, out_size);
  return false;
}

bool DistributedStore::lustre_contains(const BlockId &id) const {
  if (agg_lustre_) return agg_lustre_->contains(id);
  if (lustre_) return lustre_->contains(id);
  return false;
}

bool DistributedStore::is_prefix(const BlockId &id) const {
  std::shared_lock lock(prefix_mutex_);
  return prefix_registry_.count(id) > 0;
}

// ============================================================================
// Infrastructure: Barrier, Metadata Sync, Stats
// ============================================================================

void DistributedStore::barrier() {
#ifdef USE_MPI
  MPI_Barrier(comm_);
#endif
}

void DistributedStore::sync_metadata() {
#ifdef USE_MPI
  sync_prefix_registry();
  
  // Sync global index
  std::vector<BlockId> keys = gpu_->global_index_.keys();
  std::vector<char> send_buf;
  send_buf.insert(send_buf.end(), sizeof(uint32_t), 0);
  uint32_t actual_count = 0;
  for (const auto& key : keys) {
    auto loc = gpu_->global_index_.get(key);
    if (loc && loc->node_id == rank_) {
      uint32_t klen = key.size();
      send_buf.insert(send_buf.end(), (char*)&klen, (char*)&klen + sizeof(klen));
      send_buf.insert(send_buf.end(), key.begin(), key.end());
      send_buf.insert(send_buf.end(), (char*)&(*loc), (char*)&(*loc) + sizeof(BlockLocation));
      actual_count++;
    }
  }
  memcpy(send_buf.data(), &actual_count, sizeof(actual_count));

  int send_size = send_buf.size();
  std::vector<int> recv_sizes(world_size_);
  MPI_Allgather(&send_size, 1, MPI_INT, recv_sizes.data(), 1, MPI_INT, comm_);
  std::vector<int> displs(world_size_, 0);
  int total = 0;
  for(int i=0; i<world_size_; i++) { displs[i]=total; total+=recv_sizes[i]; }
  std::vector<char> recv_buf(total);
  MPI_Allgatherv(send_buf.data(), send_size, MPI_CHAR, recv_buf.data(), recv_sizes.data(), displs.data(), MPI_CHAR, comm_);

  const char* ptr = recv_buf.data();
  for(int r=0; r<world_size_; r++) {
    if(r==rank_) { ptr += recv_sizes[r]; continue; }
    uint32_t cnt; memcpy(&cnt, ptr, sizeof(cnt)); ptr+=sizeof(cnt);
    for(uint32_t i=0; i<cnt; i++) {
      uint32_t klen; memcpy(&klen, ptr, sizeof(klen)); ptr+=sizeof(klen);
      BlockId id(ptr, ptr+klen); ptr+=klen;
      BlockLocation loc; memcpy(&loc, ptr, sizeof(loc)); ptr+=sizeof(loc);
      gpu_->global_index_.put(id, loc);
      if(cfg_.dedup_enabled) global_dedup_.put(id, true);
    }
  }
  if(rank_==0) printf("[Sync] Total global blocks: %zu\n", gpu_->global_index_.size());
#endif
  barrier();
}

DistributedStore::Stats DistributedStore::get_stats() {
  Stats s{};
  s.local_gpu_used = gpu_ ? gpu_->used_bytes() : 0;
  s.local_dram_used = dram_ ? dram_->used_bytes() : 0;
  s.local_gpu_hits = local_gpu_hits_.load();
  s.local_dram_hits = local_dram_hits_.load();
  s.remote_gpu_hits = remote_gpu_hits_.load();
  s.remote_dram_hits = remote_dram_hits_.load();
  s.lustre_hits = lustre_hits_.load();
  s.misses = misses_.load();
  s.dedup_hits = dedup_hits_.load();
  s.dedup_bytes_saved = dedup_bytes_saved_.load();
  s.gpu_evictions = gpu_evictions_.load();
  s.dram_evictions = dram_evictions_.load();
  s.prefix_blocks_protected = prefix_blocks_protected_.load();
  s.promotions_to_local = promotions_to_local_.load();
  s.compression_savings = compression_savings_.load();
  s.total_blocks = global_dedup_.size();

  {
    std::shared_lock lock(prefix_mutex_);
    s.prefix_blocks = prefix_registry_.size();
  }

#ifdef USE_MPI
  MPI_Allreduce(&s.local_gpu_used, &s.cluster_gpu_used, 1, MPI_UNSIGNED_LONG,
                MPI_SUM, comm_);
  MPI_Allreduce(&s.local_dram_used, &s.cluster_dram_used, 1, MPI_UNSIGNED_LONG,
                MPI_SUM, comm_);
#else
  s.cluster_gpu_used = s.local_gpu_used;
  s.cluster_dram_used = s.local_dram_used;
#endif

  return s;
}

} // namespace distributed
} // namespace cascade

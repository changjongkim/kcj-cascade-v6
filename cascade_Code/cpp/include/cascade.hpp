/**
 * Cascade KV Cache - High-Performance C++ Core V6
 * 
 * Features:
 *   - LRU eviction with semantic awareness (prefix preservation)
 *   - Tier promotion/demotion (Lustre→SHM→GPU on read)
 *   - Free-list memory management for GPU and SHM pools
 *   - 32 CUDA streams + pinned buffers for GPU transfers
 *   - SSE2 streaming stores for SHM I/O
 *   - Content-addressed deduplication (SHA256)
 *   - Async prefetch pipeline (Lustre→SHM background loading)
 *   - INT8 KV compression (2× storage savings)
 *   - Aggregated Lustre I/O (multi-block files)
 * 
 * Target: 25+ GB/s on PCIe Gen4, near-hardware limits
 * 
 * Author: SC26 Cascade Team
 */

#pragma once

#include <string>
#include <vector>
#include <list>
#include <memory>
#include <atomic>
#include <shared_mutex>
#include <unordered_map>
#include <functional>
#include <optional>
#include <cstdint>
#include <cstring>
#include <array>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <deque>

namespace cascade {

// ============================================================================
// Configuration
// ============================================================================

struct CascadeConfig {
    // GPU tier
    size_t gpu_capacity_bytes = 32ULL * 1024 * 1024 * 1024;  // 32GB
    int gpu_device_id = 0;
    bool use_gpu = true;
    
    // SHM tier (DRAM)
    size_t shm_capacity_bytes = 64ULL * 1024 * 1024 * 1024;  // 64GB
    std::string shm_path = "/dev/shm/cascade";
    
    // Lustre tier
    std::string lustre_path = "/pscratch/sd/s/sgkim/cascade_store";
    size_t lustre_stripe_size = 1024 * 1024;  // 1MB
    int lustre_stripe_count = 16;
    
    // Features
    bool dedup_enabled = true;
    bool compression_enabled = true;
    int num_io_threads = 8;
    
    // Eviction policy
    bool semantic_eviction = true;  // Preserve prefix blocks during eviction
    bool promotion_enabled = true;  // Promote data to higher tiers on read
    
    // Async Prefetch
    bool prefetch_enabled = true;   // Background Lustre→SHM prefetching
    int prefetch_threads = 2;       // Number of prefetch worker threads
    size_t prefetch_queue_size = 64; // Max outstanding prefetch requests
    
    // KV Compression
    bool kv_compression = false;    // INT8 quantization for SHM/Lustre storage
    
    // Aggregated Lustre I/O
    bool aggregated_lustre = false; // Multi-block files instead of per-block
    size_t agg_file_size = 256ULL * 1024 * 1024;  // 256MB per aggregate file
};

// ============================================================================
// Block ID (32-byte SHA256 prefix as string)
// ============================================================================

using BlockId = std::string;

BlockId compute_block_id(const uint8_t* data, size_t size);

// ============================================================================
// LRU Sharded Index (256 shards, per-shard LRU tracking)
// ============================================================================

template<typename V>
class ShardedIndex {
public:
    static constexpr size_t NUM_SHARDS = 256;
    
    ShardedIndex();
    ~ShardedIndex();
    
    bool put(const BlockId& key, V value, size_t size = 0);
    std::optional<V> get(const BlockId& key) const;
    bool remove(const BlockId& key);
    bool contains(const BlockId& key) const;
    
    // LRU support — touch moves entry to front (most recently used)
    std::optional<V> get_and_touch(const BlockId& key);
    
    // Get the LRU victim (least recently used entry)
    // Returns (key, value, size). Caller decides whether to evict.
    struct LRUEntry {
        BlockId key;
        V value;
        size_t size;
    };
    std::optional<LRUEntry> get_lru_victim() const;
    
    // Get the LRU victim, skipping keys in the skip set (for semantic eviction)
    std::optional<LRUEntry> get_lru_victim_skip(
        const std::function<bool(const BlockId&)>& should_skip) const;
    
    size_t total_size() const;
    size_t total_count() const;
    void clear();
    
private:
    struct LRUNode {
        BlockId key;
        V value;
        size_t size;
    };
    
    struct Shard {
        mutable std::shared_mutex mutex;
        
        // LRU list: front = most recently used, back = least recently used
        std::list<LRUNode> lru_list;
        
        // Fast lookup: key → iterator into lru_list
        std::unordered_map<BlockId, typename std::list<LRUNode>::iterator> index;
        
        std::atomic<size_t> total_size{0};
    };
    
    std::array<Shard, NUM_SHARDS> shards_;
    
    size_t get_shard_id(const BlockId& key) const {
        return std::hash<BlockId>{}(key) % NUM_SHARDS;
    }
};

// ============================================================================
// GPU Backend (CUDA + Memory Pool + Multi-Stream + Free List)
// ============================================================================

class GPUMemoryPool;  // Forward declaration

class GPUBackend {
public:
    GPUBackend(size_t capacity_bytes, int device_id = 0);
    ~GPUBackend();
    
    bool put(const BlockId& id, const uint8_t* data, size_t size);
    bool get(const BlockId& id, uint8_t* out_data, size_t* out_size);
    bool remove(const BlockId& id);
    bool contains(const BlockId& id) const;
    
    // LRU-aware get: touches entry to mark as recently used
    bool get_and_touch(const BlockId& id, uint8_t* out_data, size_t* out_size);
    
    // Eviction support
    struct EvictedBlock {
        BlockId id;
        std::vector<uint8_t> data;
        size_t size;
    };
    
    // Evict LRU block, returns its data for demotion to lower tier
    std::optional<EvictedBlock> evict_lru();
    
    // Evict LRU block, skipping prefix blocks (semantic eviction)
    std::optional<EvictedBlock> evict_lru_semantic(
        const std::function<bool(const BlockId&)>& is_prefix);
    
    // Evict enough blocks to free at least `needed_bytes`
    std::vector<EvictedBlock> evict_for_space(
        size_t needed_bytes,
        const std::function<bool(const BlockId&)>& is_prefix = nullptr);
    
    size_t used_bytes() const { return used_.load(); }
    size_t capacity() const { return capacity_; }
    
    void clear();
    void sync_all();  // Sync all pending async transfers
    
    // Remote access support
    uint8_t* get_base_ptr() const;
    size_t get_offset(const BlockId& id) const;
    
private:
    size_t capacity_;
    int device_id_;
    std::atomic<size_t> used_{0};
    
    // Memory pool (pre-allocated GPU memory with free list)
    std::unique_ptr<GPUMemoryPool> memory_pool_;
    
    // 32 pinned buffers for parallel transfers (one per thread)
    static constexpr int NUM_PINNED_BUFFERS = 32;
    void* pinned_buffers_[32] = {nullptr};
    void* pinned_buffer_ = nullptr;  // Legacy
    size_t pinned_size_ = 8 * 1024 * 1024;  // 8MB per staging buffer
    
    // 32 CUDA streams - one per thread = ZERO CONTENTION
    static constexpr int NUM_STREAMS = 32;
    void* cuda_streams_[32] = {nullptr};
    std::atomic<int> current_stream_{0};
    
    // Index: block_id -> (gpu_ptr, size) with LRU tracking
    struct GPUBlock {
        void* ptr;
        size_t size;
    };
    ShardedIndex<GPUBlock> index_;
    
    bool init_cuda();
    void* alloc_gpu(size_t size);
    void free_gpu(void* ptr, size_t size);  // Now actually frees via free list
    void copy_h2d_async(void* dst, const void* src, size_t size, int stream);
    void copy_d2h_async(void* dst, const void* src, size_t size, int stream);
    void sync_stream(int stream);
    
    // Internal: read GPU memory to host buffer
    bool read_block_to_host(const GPUBlock& block, uint8_t* out_data);
};

// ============================================================================
// SHM Backend (mmap + Free List + LRU)
// ============================================================================

class ShmBackend {
public:
    ShmBackend(size_t capacity_bytes, const std::string& path = "/dev/shm/cascade");
    ~ShmBackend();
    
    bool put(const BlockId& id, const uint8_t* data, size_t size);
    bool get(const BlockId& id, uint8_t* out_data, size_t* out_size);
    bool remove(const BlockId& id);
    bool contains(const BlockId& id) const;
    
    // LRU-aware get
    bool get_and_touch(const BlockId& id, uint8_t* out_data, size_t* out_size);
    
    // Eviction support
    struct EvictedBlock {
        BlockId id;
        std::vector<uint8_t> data;
        size_t size;
    };
    
    std::optional<EvictedBlock> evict_lru();
    std::optional<EvictedBlock> evict_lru_semantic(
        const std::function<bool(const BlockId&)>& is_prefix);
    std::vector<EvictedBlock> evict_for_space(
        size_t needed_bytes,
        const std::function<bool(const BlockId&)>& is_prefix = nullptr);
    
    size_t used_bytes() const { return used_.load(); }
    size_t capacity() const { return capacity_; }
    
    void clear();
    
private:
    size_t capacity_;
    std::string path_;
    std::atomic<size_t> used_{0};
    
    // Memory-mapped region
    void* mmap_base_ = nullptr;
    size_t mmap_size_ = 0;
    std::atomic<size_t> write_offset_{0};
    
    // Free list: (offset, size) pairs available for reuse
    struct FreeBlock {
        size_t offset;
        size_t size;
    };
    std::list<FreeBlock> free_list_;
    mutable std::mutex free_list_mutex_;
    
    // Allocate from free list or bump pointer
    size_t allocate(size_t size);
    // Return memory to free list
    void deallocate(size_t offset, size_t size);
    
    // Index with LRU
    struct ShmBlock {
        size_t offset;
        size_t size;
    };
    ShardedIndex<ShmBlock> index_;
    
    // Internal: read SHM memory (SSE2 optimized)
    void read_block(const ShmBlock& block, uint8_t* out_data);
    void write_block(uint8_t* dst, const uint8_t* src, size_t size);
};

// ============================================================================
// Lustre Backend (file-based cold storage)
// ============================================================================

class LustreBackend {
public:
    LustreBackend(const std::string& path, size_t stripe_size = 1024*1024, int stripe_count = 16);
    ~LustreBackend();
    
    bool put(const BlockId& id, const uint8_t* data, size_t size);
    bool get(const BlockId& id, uint8_t* out_data, size_t* out_size);
    bool remove(const BlockId& id);
    bool contains(const BlockId& id) const;
    
    void flush();
    
    // For aggregated backend to access path utilities
    std::string get_base_path() const { return base_path_; }
    
private:
    std::string base_path_;
    size_t stripe_size_;
    int stripe_count_;
    
    std::string block_path(const BlockId& id) const;
};

// ============================================================================
// Aggregated Lustre Backend (multi-block files)
//
// Instead of 1 file per block (high metadata overhead on Lustre),
// appends blocks into large aggregate files (~256MB each).
// An in-memory index maps BlockId → (file_id, offset, size).
// ============================================================================

class AggregatedLustreBackend {
public:
    AggregatedLustreBackend(const std::string& path, size_t max_file_size = 256ULL*1024*1024,
                            size_t stripe_size = 4*1024*1024, int stripe_count = 16);
    ~AggregatedLustreBackend();
    
    bool put(const BlockId& id, const uint8_t* data, size_t size);
    bool get(const BlockId& id, uint8_t* out_data, size_t* out_size);
    bool contains(const BlockId& id) const;
    std::vector<BlockId> list_blocks() const;
    
    void flush();
    
private:
    struct BlockLocation {
        uint32_t file_id;
        size_t offset;
        size_t size;
    };
    
    std::string base_path_;
    size_t max_file_size_;
    size_t stripe_size_;
    int stripe_count_;
    
    // Current write file
    int current_fd_ = -1;
    uint32_t current_file_id_ = 0;
    size_t current_offset_ = 0;
    std::mutex write_mutex_;
    
    // Block index
    mutable std::shared_mutex index_mutex_;
    std::unordered_map<BlockId, BlockLocation> index_;
    
    std::string file_path(uint32_t file_id) const;
    bool open_new_file();
};

// ============================================================================
// KV Compression (INT8 quantization)
//
// FP16 KV cache → INT8 with per-channel scale factors.
// Achieves ~2× storage reduction with minimal accuracy loss.
// Format: [scale_fp32 (4 bytes)] [zero_point_int8 (1 byte)] [int8_data...]
// ============================================================================

struct CompressionMeta {
    float scale;
    int8_t zero_point;
};

class KVCompressor {
public:
    // Compress FP16 data to INT8
    // Returns compressed buffer and metadata
    static std::vector<uint8_t> compress(const uint8_t* data, size_t size, CompressionMeta& meta);
    
    // Decompress INT8 back to FP16
    // out_data must be pre-allocated with original size
    static bool decompress(const uint8_t* compressed, size_t compressed_size,
                           const CompressionMeta& meta, uint8_t* out_data, size_t original_size);
    
    // Compressed size for given original size (FP16 elements → INT8, halved)
    static size_t compressed_size(size_t original_size) {
        return sizeof(CompressionMeta) + original_size / 2;
    }
    
    // Original size from compressed size
    static size_t original_size(size_t compressed_size) {
        return (compressed_size - sizeof(CompressionMeta)) * 2;
    }
};

// ============================================================================
// Async Prefetch Pipeline
//
// Background thread(s) load blocks from Lustre into SHM ahead of time.
// Uses an access-frequency counter to predict which blocks to prefetch.
// ============================================================================

class PrefetchPipeline {
public:
    PrefetchPipeline(LustreBackend* lustre, AggregatedLustreBackend* agg_lustre,
                     ShmBackend* shm, int num_threads = 2, size_t queue_size = 64);
    ~PrefetchPipeline();
    
    // Submit a block for prefetching (non-blocking)
    void submit(const BlockId& id, size_t expected_size);
    
    // Record an access (for frequency tracking)
    void record_access(const BlockId& id);
    
    // Get prefetch stats
    struct Stats {
        size_t submitted;
        size_t completed;
        size_t skipped;  // Already in SHM
    };
    Stats get_stats() const;
    
    void stop();
    
private:
    struct PrefetchRequest {
        BlockId id;
        size_t expected_size;
    };
    
    LustreBackend* lustre_;
    AggregatedLustreBackend* agg_lustre_;
    ShmBackend* shm_;
    
    std::vector<std::thread> workers_;
    std::queue<PrefetchRequest> queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> running_{true};
    size_t max_queue_size_;
    
    // Access frequency for predictive prefetch
    mutable std::mutex freq_mutex_;
    std::unordered_map<BlockId, uint32_t> access_freq_;
    
    // Stats
    std::atomic<size_t> submitted_{0};
    std::atomic<size_t> completed_{0};
    std::atomic<size_t> skipped_{0};
    
    void worker_loop();
};

// ============================================================================
// CascadeStore - Main 3-Tier Store with LRU + Promotion + Semantic Eviction
// ============================================================================

class CascadeStore {
public:
    explicit CascadeStore(const CascadeConfig& config);
    ~CascadeStore();
    
    // Main API
    bool put(const BlockId& id, const uint8_t* data, size_t size, bool is_prefix = false);
    bool get(const BlockId& id, uint8_t* out_data, size_t* out_size);
    bool contains(const BlockId& id) const;
    
    // Batch API
    size_t put_batch(const std::vector<BlockId>& ids, 
                     const std::vector<const uint8_t*>& data,
                     const std::vector<size_t>& sizes);
    size_t get_batch(const std::vector<BlockId>& ids,
                     std::vector<uint8_t*>& out_data,
                     std::vector<size_t>& out_sizes);
    
    // Stats
    struct Stats {
        size_t gpu_used;
        size_t shm_used;
        size_t gpu_hits;
        size_t shm_hits;
        size_t lustre_hits;
        size_t misses;
        size_t dedup_hits;
        size_t gpu_evictions;
        size_t shm_evictions;
        size_t promotions_to_gpu;
        size_t promotions_to_shm;
        // V6: new stats
        size_t prefetch_completed;
        size_t compression_savings_bytes;
        size_t shm_puts;
        size_t lustre_puts;
    };
    Stats get_stats() const;
    
    void clear();
    void flush();
    
private:
    CascadeConfig config_;
    
    std::unique_ptr<GPUBackend> gpu_;
    std::unique_ptr<ShmBackend> shm_;
    std::unique_ptr<LustreBackend> lustre_;
    std::unique_ptr<AggregatedLustreBackend> agg_lustre_;  // V6
    std::unique_ptr<PrefetchPipeline> prefetcher_;          // V6
    
    // Dedup tracking
    ShardedIndex<bool> known_blocks_;
    ShardedIndex<bool> prefix_blocks_;
    
    // Stats
    mutable std::atomic<size_t> gpu_hits_{0};
    mutable std::atomic<size_t> shm_hits_{0};
    mutable std::atomic<size_t> lustre_hits_{0};
    mutable std::atomic<size_t> misses_{0};
    mutable std::atomic<size_t> dedup_hits_{0};
    mutable std::atomic<size_t> gpu_evictions_{0};
    mutable std::atomic<size_t> shm_evictions_{0};
    mutable std::atomic<size_t> promotions_to_gpu_{0};
    mutable std::atomic<size_t> promotions_to_shm_{0};
    mutable std::atomic<size_t> shm_puts_{0};
    mutable std::atomic<size_t> lustre_puts_{0};
    mutable std::atomic<size_t> compression_savings_{0};
    
    // Internal helpers
    bool is_prefix_block(const BlockId& id) const;
    
    bool evict_gpu_to_shm(size_t needed_bytes);
    bool evict_shm_to_lustre(size_t needed_bytes);
    
    bool promote_to_gpu(const BlockId& id, const uint8_t* data, size_t size, bool is_prefix);
    bool promote_to_shm(const BlockId& id, const uint8_t* data, size_t size);
    
    // V6: Lustre write helper (routes to aggregated or per-file)
    bool lustre_put(const BlockId& id, const uint8_t* data, size_t size);
    bool lustre_get(const BlockId& id, uint8_t* out_data, size_t* out_size);
    bool lustre_contains(const BlockId& id) const;
};

}  // namespace cascade

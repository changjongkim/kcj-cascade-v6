import sys
import os
import argparse
import unittest.mock as mock
import numpy as np
import torch
import h5py

# Add benchmark directory to sys.path to import adapters
REPO_ROOT = "/pscratch/sd/s/sgkim/kcj/Cascade-kcj"
sys.path.append(os.path.join(REPO_ROOT, "benchmark"))

def patch_makani_dataloader(adapter):
    """
    Patch MultifilesDataset to use the storage adapter instead of direct h5py calls.
    """
    from makani.utils.dataloaders import data_loader_multifiles
    
    original_get_data = data_loader_multifiles.MultifilesDataset._get_data
    
    def patched_get_data(self, global_idx, offset_start, offset_end, target=False):
        # Implementation of _get_data using adapter
        start_x = self.read_anchor[0]
        end_x = start_x + self.read_shape[0]
        start_y = self.read_anchor[1]
        end_y = start_y + self.read_shape[1]

        data_list = []
        for offset_idx in range(offset_start, offset_end):
            file_idx, local_idx = self._get_indices(global_idx + self.dt * offset_idx)
            
            # Construct a unique key for the specific sample
            filename = os.path.basename(self.files_paths[file_idx])
            key = f"{filename}_{local_idx}"
            
            # USE ZERO-COPY get_raw
            if hasattr(adapter, 'get_raw'):
                payload = adapter.get_raw(key)
            else:
                res = adapter.get(key)
                payload = res[0] + res[1] if res else None
            
            if payload is not None:
                # payload is an uint8 view of the shared buffer
                # Metadata: 20 channels, 720x1440. 
                # c = len(payload) // (4 * 720 * 1440)
                # Optimization: direct view as float32
                full_array = payload.view(np.float32).reshape(1, -1, 720, 1440)
                
                data = full_array[:, self.in_channels_sorted, start_x:end_x, start_y:end_y]
                
                if not self.in_channels_is_sorted:
                    data = data[:, self.in_channels_unsort, :, :]
            else:
                # Fallback to direct read
                if self.files[file_idx] is None:
                    self._open_file(file_idx)
                data = self.files[file_idx][local_idx : local_idx + 1, self.in_channels_sorted, start_x:end_x, start_y:end_y]
                
                if not self.in_channels_is_sorted:
                    data = data[:, self.in_channels_unsort, :, :]

            data_list.append(data)

        data = np.concatenate(data_list, axis=0)
        
        # Apply normalization logic from original Makani
        if self.normalize:
            if target:
                data = (data - self.out_bias) / self.out_scale
            else:
                data = (data - self.in_bias) / self.in_scale
        
        return data

    # Monkey patch the method
    data_loader_multifiles.MultifilesDataset._get_data = patched_get_data
    if int(os.environ.get("SLURM_PROCID", "0")) == 0:
        print(f"[Proxy] Patched Makani MultifilesDataset._get_data with {type(adapter).__name__}")

def ingest_to_storage(adapter, data_dir):
    """
    Ingest data from HDF5 to the storage system.
    """
    import glob
    train_files = glob.glob(os.path.join(data_dir, "train", "*.h5"))
    val_files = glob.glob(os.path.join(data_dir, "test", "*.h5"))
    all_files = train_files + val_files
    
    world_size = int(os.environ.get('SLURM_NTASKS', 1))
    rank = int(os.environ.get('SLURM_PROCID', 0))
    
    if rank == 0:
        print(f"[Proxy] Starting Ingestion of {len(all_files)} files into {type(adapter).__name__}...")
    
    num_files = len(all_files)
    num_files_local = num_files // world_size
    start_fidx = rank * num_files_local
    end_fidx = num_files if rank == world_size - 1 else start_fidx + num_files_local
    
    for count, fpath in enumerate(all_files):
        if count < start_fidx or count >= end_fidx:
            continue
            
        basename = os.path.basename(fpath)
        try:
            with h5py.File(fpath, "r") as f:
                dset = f["fields"]
                num_samples = dset.shape[0]
                
                for local_idx in range(num_samples):
                    key = f"{basename}_{local_idx}"
                    if not adapter.contains(key):
                        payload = dset[local_idx:local_idx+1].tobytes()
                        # Split payload for adapter compatibility
                        mid = len(payload) // 2
                        adapter.put(key, payload[:mid], payload[mid:])
        except Exception as e:
            if rank == 0:
                print(f"[Proxy] Warning: Failed to ingest {basename}: {e}")
                    
    adapter.flush()
    adapter.barrier()
    if rank == 0:
        print(f"[Proxy] Ingestion complete.")

def main():
    # DO NOT use makani's default parser here, it consumes args we need to pass down
    parser = argparse.ArgumentParser(description="Makani Proxy Launcher", add_help=False)
    
    # Storage system flags
    parser.add_argument("--system", type=str, default="hdf5", choices=["hdf5", "cascade", "lmcache_disk", "lmcache_redis", "redis", "pdc"])
    parser.add_argument("--data_path", type=str, required=True, help="Base path to Makani data")
    parser.add_argument("--cascade_lustre_path", type=str, default="/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/cascade_makani")
    parser.add_argument("--storage_path", type=str, default="/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/tmp/storage_makani")
    parser.add_argument("--redis_host", type=str, default="localhost")
    parser.add_argument("--redis_port", type=int, default=6379)
    parser.add_argument("--aggregated_io", action="store_true", default=True)
    parser.add_argument("--no_aggregated_io", action="store_false", dest="aggregated_io")
    parser.add_argument("--ingest", action="store_true", help="Perform data ingestion into storage system")
    
    # Parse only the args we know
    args, unknown = parser.parse_known_args()

    # 1. SETUP PATHS FIRST
    makani_path = os.path.join(REPO_ROOT, "ml_workspace/makani")
    if makani_path not in sys.path:
        sys.path.insert(0, makani_path)

    # 2. CONFIGURE WARP IMMEDIATELY
    try:
        import warp as wp
        # Attempt both env var and config object
        os.environ["WARP_DEVICE_MEM_POOL_SIZE"] = str(2 * 1024**3)
        wp.config.device_mem_pool_size = 2 * 1024**3
        print(f"[Proxy] WARP CONFIG: Pool size set to 2GB", flush=True)
    except ImportError:
        pass
    except Exception as e:
        print(f"[Proxy] WARP CONFIG ERROR: {e}", flush=True)

    # 3. REGISTER MODELS
    # We do this before initializing the adapter to avoid race conditions with imports
    try:
        from makani.models.model_registry import register_model
        
        # Try to register FCN3
        try:
            from makani.models.networks.fourcastnet3 import AtmoSphericNeuralOperatorNet
            register_model(AtmoSphericNeuralOperatorNet, "FCN3")
            print("[Proxy] Registered FCN3 model (AtmoSphericNeuralOperatorNet).", flush=True)
        except Exception as e:
             print(f"[Proxy] FCN3 registration failed: {e}", flush=True)

        # Try to register SFNO
        try:
            from makani.models.networks.sfnonet import SphericalFourierNeuralOperatorNet
            register_model(SphericalFourierNeuralOperatorNet, "SFNO")
            print("[Proxy] Registered SFNO model (SphericalFourierNeuralOperatorNet).", flush=True)
        except Exception as e:
             print(f"[Proxy] SFNO registration failed: {e}", flush=True)
    except Exception as e:
        print(f"[Proxy] Model registry access failed: {e}", flush=True)

    # 4. Initialize Storage Adapter
    adapter = None
    if args.system == 'cascade':
        from adapters.cascade_adapter_ml import CascadeAdapter
        class MLProxyAdapter(CascadeAdapter):
            def initialize(self):
                import cascade_cpp
                world_size = int(os.environ.get("SLURM_NTASKS", "1"))
                if world_size > 1:
                    cfg = cascade_cpp.DistributedConfig()
                    cfg.gpu_capacity_per_device = 0
                    cfg.dram_capacity = 1 * 1024**3
                    cfg.num_gpus_per_node = 4
                    cfg.kv_compression = False
                    cfg.locality_aware = True
                    cfg.prefix_replication = False
                    cfg.aggregated_lustre = False
                    cfg.lustre_path = args.cascade_lustre_path
                    self.store = cascade_cpp.DistributedStore(cfg)
                    self._initialized = True
                    return True
                return super().initialize()
        adapter = MLProxyAdapter({"lustre_path": args.cascade_lustre_path})
    elif args.system == 'pdc':
        from adapters.pdc_adapter import PDCAdapter
        adapter = PDCAdapter({"storage_path": args.storage_path})
    elif args.system == 'lmcache_disk':
        from adapters.lmcache_adapter import LMCacheAdapter
        adapter = LMCacheAdapter({"storage_path": args.storage_path})
    elif args.system == 'lmcache_redis' or args.system == 'redis':
        from adapters.redis_adapter import RedisAdapter
        adapter = RedisAdapter({"host": args.redis_host, "port": args.redis_port})
    if adapter:
        adapter.initialize()
        
        # Perform ingestion if requested
        if args.ingest:
            ingest_to_storage(adapter, args.data_path)
            
        # Patch Makani dataloader
        patch_makani_dataloader(adapter)
    
    # Prepare to call original Makani train
    sys.argv = [sys.argv[0]] + unknown
    
    import runpy
    runpy.run_path(os.path.join(makani_path, "makani/train.py"), run_name="__main__")

if __name__ == "__main__":
    main()

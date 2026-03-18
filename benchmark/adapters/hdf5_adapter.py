# benchmark/adapters/hdf5_adapter.py
"""
Adapter for HDF5 storage (baseline comparison).
Supports Independent and Collective I/O.
"""
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

try:
    from .base import StorageAdapter
except ImportError:
    from benchmark.adapters.base import StorageAdapter


class HDF5Adapter(StorageAdapter):
    """
    Base HDF5-based KV cache storage.
    """
    
    def __init__(self, name: str = "HDF5", config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.h5file = None
        base_path = config.get("file_path", f"/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/hdf5_store/kv_cache_{name.lower()}.h5")
        
        # MPI info
        self.rank = int(os.environ.get('SLURM_PROCID', 0))
        self.world_size = int(os.environ.get('SLURM_NTASKS', 1))
        
        self.use_mpi = config.get("use_mpi", False)
        self.file_per_rank = config.get("file_per_rank", not self.use_mpi) # Default True for Indep
        
        if self.file_per_rank:
            p = Path(base_path)
            self.file_path = str(p.parent / f"{p.stem}_r{self.rank}{p.suffix}")
        else:
            self.file_path = base_path

        self.compression = config.get("compression", None)
        self.compression_level = config.get("compression_level", 1)
        
        # Stats
        self._reads = 0
        self._writes = 0
    
    def initialize(self) -> bool:
        try:
            os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
            import h5py
            
            # Create directory
            Path(self.file_path).parent.mkdir(parents=True, exist_ok=True)
            
            if self.use_mpi:
                try:
                    import mpi4py
                    from mpi4py import MPI
                    
                    if not MPI.Is_initialized():
                        # We cannot initialize it here reliably on Perlmutter due to conda OFI conflicts
                        print(f"[{self.name}] MPI not initialized in python but needed for collective I/O", flush=True)
                        return False

                    print(f"[{self.name}] Rank {self.rank} Collective Init. h5py: {h5py.__version__}, MPI: {h5py.get_config().mpi}", flush=True)
                    # Collective I/O requires a shared file
                    self.h5file = h5py.File(self.file_path, 'a', driver='mpio', comm=MPI.COMM_WORLD)
                    
                    # Create groups if needed (collective operation)
                    if 'keys' not in self.h5file:
                        self.h5file.create_group('keys')
                    if 'values' not in self.h5file:
                        self.h5file.create_group('values')
                    
                    self._initialized = True
                    return True
                except Exception as e:
                    print(f"[{self.name}] MPI/Collective I/O error: {e}")
                    return False
            else:
                # Independent I/O
                self.h5file = h5py.File(self.file_path, 'a', libver='latest')
                print(f"[{self.name}] Initialized Rank {self.rank}/{self.world_size}, file: {self.file_path}", flush=True)
                
                # Create groups if needed
                if 'keys' not in self.h5file:
                    self.h5file.create_group('keys')
                if 'values' not in self.h5file:
                    self.h5file.create_group('values')
                
                self._initialized = True
                return True
        except ImportError as e:
            print(f"[{self.name}] Missing dependency: {e}", flush=True)
            return False
        except Exception as e:
            print(f"[{self.name}] Init error: {e}", flush=True)
            return False
    
    def put(self, block_id: str, key_data: bytes, value_data: bytes) -> bool:
        if not self._initialized:
            return False
        
        if self.h5file is None:
            if not self.initialize():
                return False
        
        try:
            key_arr = np.frombuffer(key_data, dtype=np.float16)
            val_arr = np.frombuffer(value_data, dtype=np.float16)
            
            dataset_name = f"{block_id}"
            
            # Note: In Collective I/O mode (use_mpi=True), ALL ranks must generally participate
            # in dataset creation. If put() is called by only one rank, this will HANG.
            # Thus, for the current benchmark pattern, file-per-rank Independent I/O is often better.
            
            if dataset_name in self.h5file['keys']:
                pass 
            else:
                kwargs = {}
                if self.compression:
                    kwargs['compression'] = self.compression
                    kwargs['compression_opts'] = self.compression_level
                
                self.h5file['keys'].create_dataset(dataset_name, data=key_arr, **kwargs)
                self.h5file['values'].create_dataset(dataset_name, data=val_arr, **kwargs)

            self.h5file.flush()
            
            self._writes += 1
            return True
            
        except Exception as e:
            print(f"[{self.name}] Put error: {e}")
            return False
    
    def get(self, block_id: str) -> Optional[tuple]:
        if not self._initialized:
            return None
            
        if self.h5file is None:
            if not self.initialize():
                return None
        
        try:
            dataset_name = f"{block_id}"
            
            # 1. Try local file first
            if self.h5file and dataset_name in self.h5file['keys']:
                key_arr = self.h5file['keys'][dataset_name][:]
                val_arr = self.h5file['values'][dataset_name][:]
                self._reads += 1
                return (key_arr.tobytes(), val_arr.tobytes())
            
            # 2. If not found and using file-per-rank, search other files
            if self.file_per_rank:
                p = Path(self.file_path)
                target_ranks = list(range(self.world_size))
                
                for r in target_ranks:
                    # If we have our own file open and this is our rank, skip re-opening
                    if r == self.rank and self.h5file is not None: continue
                    
                    other_path = str(p.parent / f"{p.name.replace(f'_r{self.rank}', f'_r{r}')}")
                    
                    if not os.path.exists(other_path):
                        pass # Avoid expensive ls -f on Lustre
                    
                    if os.path.exists(other_path):
                        try:
                            # Open in SWMR mode for reading potentially active files
                            import h5py
                            with h5py.File(other_path, 'r', libver='latest', swmr=True) as f:
                                if 'keys' in f and dataset_name in f['keys']:
                                    key_arr = f['keys'][dataset_name][:]
                                    val_arr = f['values'][dataset_name][:]
                                    self._reads += 1
                                    return (key_arr.tobytes(), val_arr.tobytes())
                        except Exception:
                            continue
            
            return None
        except Exception:
            return None


    
    def contains(self, block_id: str) -> bool:
        if not self._initialized:
            return False
        
        if self.h5file is None:
            if not self.initialize():
                return False
        return block_id in self.h5file['keys']
    
    def delete(self, block_id: str) -> bool:
        if not self._initialized:
            return False
        
        try:
            dataset_name = f"{block_id}"
            if dataset_name in self.h5file['keys']:
                del self.h5file['keys'][dataset_name]
                del self.h5file['values'][dataset_name]
                return True
            return False
        except:
            return False
    
    def clear(self) -> None:
        if self._initialized and self.h5file:
            # In parallel, clearing is dangerous
            if self.file_per_rank:
                for k in list(self.h5file['keys'].keys()):
                    del self.h5file['keys'][k]
                for k in list(self.h5file['values'].keys()):
                    del self.h5file['values'][k]
            else:
                if self.rank == 0:
                    for k in list(self.h5file['keys'].keys()):
                        del self.h5file['keys'][k]
                    for k in list(self.h5file['values'].keys()):
                        del self.h5file['values'][k]
            
            if self.use_mpi:
                from mpi4py import MPI
                MPI.COMM_WORLD.Barrier()
                
            self._reads = 0
            self._writes = 0
    
    def flush(self) -> None:
        if self._initialized and self.h5file:
            self.h5file.flush()
            
            # Force Lustre to sync metadata by fully closing
            if self.file_per_rank:
                try:
                    self.h5file.close()
                    self.h5file = None
                except Exception as e:
                    pass
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "reads": self._reads,
            "writes": self._writes,
            "file_size_mb": os.path.getsize(self.file_path) / 1024**2 if os.path.exists(self.file_path) else 0,
            "mode": "collective" if self.use_mpi else "independent"
        }
    
    def close(self) -> None:
        if self.h5file:
            try:
                self.h5file.close()
            except:
                pass
            self.h5file = None


class HDF5IndependentIOAdapter(HDF5Adapter):
    """HDF5 with Independent I/O"""
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        config['use_mpi'] = False
        super().__init__("HDF5-Independent", config)


class HDF5CollectiveIOAdapter(HDF5Adapter):
    """HDF5 with Collective I/O (requires MPI)"""
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        config['use_mpi'] = True
        super().__init__("HDF5-Collective", config)


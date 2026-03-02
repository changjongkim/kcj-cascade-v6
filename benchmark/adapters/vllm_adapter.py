# benchmark/adapters/vllm_adapter.py
"""
Adapter for vLLM GPU-based KV cache (baseline comparison).
"""
import time
import os
import shutil
from typing import Optional, Dict, Any

try:
    from .base import StorageAdapter
except ImportError:
    from benchmark.adapters.base import StorageAdapter

class vLLMGPUAdapter(StorageAdapter):
    """
    vLLM-style GPU KV cache.
    Simulates keeping KV tensors in GPU memory.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("vLLM-GPU", config)
        self.cache = {}
        self.device = config.get("device", "cuda:0")
        # In multi-node, we need a shared path to simulate fetching
        self.storage_path = config.get("storage_path", "/pscratch/sd/s/sgkim/kcj/Cascade-kcj/benchmark/vllm_store")
        self.use_shared = config.get("use_shared", True) # Default to true for multi-node support
        
        # Stats
        self._reads = 0
        self._writes = 0

    def initialize(self) -> bool:
        try:
            import torch
            self.has_torch = True
            if not torch.cuda.is_available():
                print("[vLLMGPUAdapter] CUDA not available, using CPU torch")
            
            if self.use_shared:
                import os
                os.makedirs(self.storage_path, exist_ok=True)
                
            self._initialized = True
            return True
        except ImportError:
            print("[vLLMGPUAdapter] PyTorch not installed, using numpy fallback")
            self.has_torch = False
            if self.use_shared:
                import os
                os.makedirs(self.storage_path, exist_ok=True)
            self._initialized = True
            return True


    def put(self, block_id: str, key_data: bytes, value_data: bytes) -> bool:
        if not self._initialized: return False
        
        try:
            import numpy as np
            data = key_data + value_data
            
            # 1. Local Memory Cache
            if self.has_torch:
                import torch
                arr = np.frombuffer(data, dtype=np.float16)
                tensor = torch.from_numpy(arr).to(self.device)
                self.cache[block_id] = tensor
            else:
                self.cache[block_id] = np.frombuffer(data, dtype=np.uint8).copy()
            
            # 2. Shared Storage Cache (for multi-node simulated fetch)
            if self.use_shared:
                path = os.path.join(self.storage_path, f"{block_id}.vllm")
                with open(path, 'wb') as f:
                    f.write(data)

            self._writes += 1
            return True
        except Exception as e:
            print(f"[vLLMGPUAdapter] Put error: {e}")
            return False

    def get(self, block_id: str) -> Optional[tuple]:
        if not self._initialized: return None
        
        try:
            # 1. Check Local Cache
            if block_id in self.cache:
                val = self.cache[block_id]
                if self.has_torch:
                    data = val.cpu().numpy().tobytes()
                else:
                    data = val.tobytes()
                self._reads += 1
                mid = len(data) // 2
                return (data[:mid], data[mid:])
            
            # 2. Check Shared Storage (Distributed Fetch simulation)
            if self.use_shared:
                path = os.path.join(self.storage_path, f"{block_id}.vllm")
                if os.path.exists(path):
                    with open(path, 'rb') as f:
                        data = f.read()
                    
                    # Also load into local cache once fetched
                    if not self.has_torch:
                        import numpy as np
                        self.cache[block_id] = np.frombuffer(data, dtype=np.uint8).copy()
                    
                    self._reads += 1
                    mid = len(data) // 2
                    return (data[:mid], data[mid:])
            
            return None
        except Exception as e:
            return None

    def contains(self, block_id: str) -> bool:
        if block_id in self.cache: return True
        if self.use_shared:
            return os.path.exists(os.path.join(self.storage_path, f"{block_id}.vllm"))
        return False

    def delete(self, block_id: str) -> bool:
        removed = False
        if block_id in self.cache:
            del self.cache[block_id]
            removed = True
        if self.use_shared:
            path = os.path.join(self.storage_path, f"{block_id}.vllm")
            if os.path.exists(path):
                os.remove(path)
                removed = True
        return removed

    def clear(self) -> None:
        self.cache.clear()
        rank = int(os.environ.get('SLURM_PROCID', 0))
        if rank == 0 and self.use_shared and os.path.exists(self.storage_path):
            import shutil
            for f in os.listdir(self.storage_path):
                try: os.remove(os.path.join(self.storage_path, f))
                except: pass
                
        if self.has_torch:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self._reads = 0
        self._writes = 0

    def flush(self) -> None:
        if self.has_torch:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "num_blocks": len(self.cache),
            "reads": self._reads,
            "writes": self._writes
        }

    def close(self) -> None:
        self.clear()
        self._initialized = False

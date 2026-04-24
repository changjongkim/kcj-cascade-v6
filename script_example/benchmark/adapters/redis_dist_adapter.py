
import os
import time
import hashlib
from typing import Optional, Dict, Any, List

try:
    from .base import StorageAdapter
except ImportError:
    from benchmark.adapters.base import StorageAdapter

class RedisDistAdapter(StorageAdapter):
    CHUNK_SIZE = 50 * 1024 * 1024

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("RedisDist", config)

        self.hosts = config.get("hosts", ["localhost"])
        self.port = config.get("port", 16379)
        self.password = config.get("password", None)
        self.clients = []
        self._num_shards = 0

    def initialize(self) -> bool:
        try:
            import redis
            self.clients = []
            for host in self.hosts:
                client = redis.Redis(
                    host=host,
                    port=self.port,
                    password=self.password,
                    socket_timeout=30,
                    socket_connect_timeout=10,
                    retry_on_timeout=True,
                )

                client.ping()
                self.clients.append(client)

            self._num_shards = len(self.clients)
            if self._num_shards > 0:
                self._initialized = True
                return True
            return False
        except Exception as e:
            print(f"[RedisDistAdapter] Initialization failed: {e}")
            return False

    def _get_client(self, block_id: str):

        idx = int(hashlib.md5(block_id.encode()).hexdigest(), 16) % self._num_shards
        return self.clients[idx]

    def put(self, block_id: str, key_data: bytes, value_data: bytes) -> bool:
        if not self._initialized: return False
        client = self._get_client(block_id)
        try:
            raw = key_data + value_data
            total = len(raw)
            if total <= self.CHUNK_SIZE:
                client.set(block_id, raw)
                client.set(f"{block_id}:meta", str(total).encode())
            else:
                n_chunks = (total + self.CHUNK_SIZE - 1) // self.CHUNK_SIZE
                pipe = client.pipeline(transaction=False)
                for i in range(n_chunks):
                    start = i * self.CHUNK_SIZE
                    end = min(start + self.CHUNK_SIZE, total)
                    pipe.set(f"{block_id}:chunk:{i}", raw[start:end])
                pipe.set(f"{block_id}:meta", f"{total}:{n_chunks}".encode())
                pipe.execute()
            return True
        except Exception as e:
            print(f"[RedisDistAdapter] Put error: {e}")
            return False

    def get(self, block_id: str) -> Optional[tuple]:
        if not self._initialized: return None
        client = self._get_client(block_id)
        try:
            meta = client.get(f"{block_id}:meta")
            if meta is None: return None

            meta_str = meta.decode()
            if ':'not in meta_str:
                val = client.get(block_id)
                if val: data = val
                else: return None
            else:
                total, n_chunks = meta_str.split(':')
                total, n_chunks = int(total), int(n_chunks)
                pipe = client.pipeline(transaction=False)
                for i in range(n_chunks):
                    pipe.get(f"{block_id}:chunk:{i}")
                chunks = pipe.execute()
                data = b''.join(chunks)

            mid = len(data) // 2
            return (data[:mid], data[mid:])
        except Exception as e:
            print(f"[RedisDistAdapter] Get error: {e}")
            return None

    def contains(self, block_id: str) -> bool:
        if not self._initialized: return False
        return self._get_client(block_id).exists(f"{block_id}:meta") > 0

    def delete(self, block_id: str) -> bool:
        if not self._initialized: return False
        client = self._get_client(block_id)
        try:
            meta = client.get(f"{block_id}:meta")
            if meta:
                meta_str = meta.decode()
                if ':'in meta_str:
                    _, n_chunks = meta_str.split(':')
                    keys = [f"{block_id}:chunk:{i}"for i in range(int(n_chunks))]
                    keys.append(f"{block_id}:meta")
                    client.delete(*keys)
                else:
                    client.delete(block_id, f"{block_id}:meta")
                return True
            return False
        except:
            return False

    def clear(self) -> None:
        if self._initialized:
            for client in self.clients:
                client.flushdb()

    def flush(self) -> None:

        pass

    def get_stats(self) -> Dict[str, Any]:
        return {
            "num_shards": self._num_shards,
            "shard_hosts": self.hosts,
            "connected": self._initialized
        }

    def close(self) -> None:
        if not self.clients: return
        for client in self.clients:
            client.close()
        self.clients = []

import torch
import threading
import collections
from typing import Dict, Optional
import time

class MockFlag:
    def __init__(self):
        self.event = threading.Event()
        self.event.set()
    def set(self):
        self.event.clear()
        t = threading.Thread(target=lambda: (time.sleep(0.001), self.event.set()))
        t.start()
    def wait(self):
        self.event.wait()
    def is_set(self):
        return self.event.is_set()

class MockGAHandle:
    def __init__(self, num_blocks: int, bytes_per_block: int):
        self.num_blocks = num_blocks
        self.bytes_per_block = bytes_per_block
        self.total_bytes = num_blocks * bytes_per_block
        self.data: Optional[torch.Tensor] = None
        self.lock = threading.Lock()
    def getNumBlocks(self) -> int:
        return self.num_blocks
    def getBytesPerBlock(self) -> int:
        return self.bytes_per_block
    def getGAddr(self) -> int:
        return id(self)
    def free_ga(self):
        with self.lock:
            del self.data
            self.data = None
    def __del__(self):
        self.free_ga()

class MockKVManager:
    def __init__(self, capacity=10000):
        self.lock = threading.Lock()
        self.uuid_to_pace: Dict[str, MockGAHandle] = collections.OrderedDict()
        self.uuid_to_tensor: Dict[str, torch.Tensor] = {}  # uuid -> tensor
        self.capacity = capacity

    def snapshot_layer_pairs(self, image_hash, layer_ids):
        """Retain immutable CPU references while the cache may evict its entries."""
        with self.lock:
            result = {}
            for layer in layer_ids:
                names = [f"{image_hash}_{layer}_{kind}" for kind in ("k", "v")]
                if any(name not in self.uuid_to_tensor for name in names):
                    return None
                result[layer] = tuple(self.uuid_to_tensor[name] for name in names)
            for layer in layer_ids:
                for kind in ("k", "v"):
                    self.uuid_to_pace.move_to_end(f"{image_hash}_{layer}_{kind}")
            return result
    def write_kv(self, tensor: torch.Tensor, uuid, non_blocking=False):
        if tensor.device.type == 'cuda':
            tensor = tensor.contiguous().cpu()
        tokens_num = tensor.shape[0]
        element_per_token = tensor.numel() // tokens_num
        bytes_per_token = tensor.element_size() * element_per_token
        pace_handle = MockGAHandle(tokens_num, bytes_per_token)
        with self.lock:
            self.uuid_to_pace[uuid] = pace_handle
            self.uuid_to_tensor[uuid] = tensor.detach().clone()
            # lru evict
            if len(self.uuid_to_pace) > self.capacity:
                old_uuid, _ = self.uuid_to_pace.popitem(last=False)
                self.uuid_to_tensor.pop(old_uuid, None)
                print(f"Evict uuid: {old_uuid}", flush=True)
        # mock async
        if non_blocking:
            flag = MockFlag()
            flag.set()
            return uuid, flag
        return uuid
    def get_kv(self, target_tensor: torch.Tensor, uuid: str, non_blocking=False):
        with self.lock:
            if uuid not in self.uuid_to_pace:
                raise KeyError(f"UUID {uuid} not found in MockKVManager")
            # lru update
            self.uuid_to_pace.move_to_end(uuid)
            pace_handle = self.uuid_to_pace[uuid]
            source_tensor = self.uuid_to_tensor[uuid]
        expected_shape = (pace_handle.getNumBlocks(),) + target_tensor.shape[1:]
        if tuple(target_tensor.shape) != expected_shape:
            raise ValueError(f"Target shape {tuple(target_tensor.shape)} != expected {expected_shape}")
        if non_blocking:
            flag = MockFlag()
            def async_copy():
                target_tensor.copy_(source_tensor)
            flag.set()
            return flag
        else:
            target_tensor.copy_(source_tensor)
            return None
    def release(self, uuid: str):
        with self.lock:
            if uuid in self.uuid_to_pace:
                self.uuid_to_pace[uuid].free_ga()
                del self.uuid_to_pace[uuid]
                self.uuid_to_tensor.pop(uuid, None)
    def release_all(self):
        with self.lock:
            for handle in self.uuid_to_pace.values():
                handle.free_ga()
            self.uuid_to_pace.clear()
            self.uuid_to_tensor.clear()

mock_kv_manager = MockKVManager()

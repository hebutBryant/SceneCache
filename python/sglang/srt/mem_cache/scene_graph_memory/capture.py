"""Request-local Qwen3-VL readout in the dedicated memory builder process."""

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

CAPTURE_ENV = "SGLANG_SCENE_GRAPH_CAPTURE_DIR"
KEY_STAGE = "qkv_projection_raw_k_pre_qk_norm_pre_mrope"
VALUE_STAGE = "qkv_projection_raw_v"


class PrefillCapture:
    """Observe a single full prefill and detach its visual rows from the GPU."""

    def __init__(self, directory, batch, layer_ids):
        self.directory = Path(directory)
        control = json.loads((self.directory / "request.json").read_text())
        self.mode = control["mode"]
        if self.mode not in {"baseline", "capture", "reuse"}:
            raise ValueError(f"Unknown graph capture mode {self.mode!r}")
        self.layer_ids = tuple(layer_ids)
        self.layers = {}
        input_ids = batch.input_ids.tolist()
        if (batch.batch_size != 1 or batch.extend_prefix_lens_cpu != [0]
                or batch.extend_seq_lens_cpu != [len(input_ids)]
                or batch.orig_seq_lens.tolist() != [len(input_ids)]):
            raise RuntimeError("Scene graph capture requires a single unchunked full prefill with no prefix hit")
        (mm_input,) = batch.mm_inputs
        if mm_input is None or len(mm_input.mm_items) != 1:
            raise ValueError("Scene graph capture requires exactly one image")
        item = mm_input.mm_items[0]
        if not item.is_image() or not item.offsets or len(item.offsets) != 1:
            raise ValueError("Scene graph capture requires exactly one image span")
        start, end = item.offsets[0]
        self.span = slice(start, end + 1)
        self.visual_rows = end - start + 1
        if not (0 < start <= end < len(input_ids) - 1):
            raise ValueError(f"Invalid visual span {item.offsets[0]}")
        if [i for i, token in enumerate(input_ids) if token == item.pad_value] != list(range(start, end + 1)):
            raise ValueError("Scheduler image placeholder IDs do not match visual span")
        if input_ids[start - 1] != mm_input.im_start_id or input_ids[end + 1] != mm_input.im_end_id:
            raise ValueError("Visual span is missing image boundary tokens")
        if batch.mrope_positions is None or list(batch.mrope_positions.shape) != [3, len(input_ids)]:
            raise ValueError("Expected one full sequence of 3-axis MRoPE positions")
        source_ids = input_ids.copy()
        source_ids[self.span] = [mm_input.im_token_id] * self.visual_rows
        (grid,) = item.image_grid_thw.tolist()
        self.metadata = {
            "mode": self.mode,
            "source_token_ids": source_ids,
            "scheduler_input_ids": input_ids,
            "image_token_id": mm_input.im_token_id,
            "source_indices": list(range(start, end + 1)),
            "source_mrope_positions": batch.mrope_positions.T.tolist(),
            "source_linear_positions": batch.positions.tolist(),
            "grid_thw": grid,
            "expected_layers": list(self.layer_ids),
            "geometry": item.scene_graph_geometry,
            "visual_rows": self.visual_rows,
        }

    @property
    def bypass(self):
        return self.mode == "capture"

    def observe_plan(self, compute_mask, recompute_info, write_info):
        layers = sorted(recompute_info)
        expected = list(self.layer_ids) if self.mode == "reuse" else []
        if layers != expected or (self.bypass and (compute_mask or write_info)):
            raise RuntimeError(f"Unexpected VLCache execution plan in {self.mode}: reused layers={layers}")
        self.metadata["reused_layers"] = layers
        self.metadata["legacy_write_layers"] = sorted(write_info)
        mask = compute_mask.get(self.layer_ids[0])
        self.metadata["recomputed_visual_rows"] = self.visual_rows if mask is None else int(mask[self.span].sum())

    def observe_embedding(self, tensor, vision):
        if tensor.ndim != 2 or tensor.shape[0] != self.visual_rows:
            raise RuntimeError(f"Vision encoder returned {tensor.shape}, expected {self.visual_rows} visual rows")
        self.metadata.update(
            embedding_shape=list(tensor.shape),
            spatial_merge_size=vision.spatial_merge_size,
            patch_size=vision.patch_size,
            deepstack_visual_indexes=list(vision.deepstack_visual_indexes),
        )

    def write_layer(self, layer_id, key, value):
        if not self.bypass or layer_id not in self.layer_ids or layer_id in self.layers:
            raise RuntimeError(f"Unexpected or duplicate KV write for layer {layer_id}")
        sequence_length = len(self.metadata["source_token_ids"])
        if key.ndim != 2 or key.shape != value.shape or key.shape[0] != sequence_length:
            raise RuntimeError(f"Layer {layer_id} did not project the entire source sequence")
        # A blocking copy gives this snapshot its own storage and completes before
        # the request or allocator releases the projection's device buffers.
        self.layers[layer_id] = tuple(
            tensor[self.span].detach().to(device="cpu", copy=True).contiguous()
            for tensor in (key, value)
        )

    def finish(self, logits):
        expected = list(self.layer_ids) if self.bypass else []
        if sorted(self.layers) != expected or "embedding_shape" not in self.metadata:
            raise RuntimeError("Prefill ended before all visual embeddings and decoder layers were captured")
        tensors = {"logits": logits.detach().to(device="cpu", copy=True).contiguous()}
        for layer_id, (key, value) in self.layers.items():
            tensors[f"{layer_id}.k"] = key
            tensors[f"{layer_id}.v"] = value
        save_file(tensors, str(self.directory / f"{self.mode}.safetensors"))
        (self.directory / f"{self.mode}.json").write_text(json.dumps(self.metadata))

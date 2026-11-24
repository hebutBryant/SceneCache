"""Native Qwen2.5-VL continuation with explicit multimodal prefix validation.

The pinned Transformers 4.57 Qwen prepare-for-generation hook removes pixels
once a cache exists. Here each direct forward receives only its local image
patches and the original full-prompt MRoPE positions for the incoming tokens.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import hashlib
from math import isfinite, prod
from time import perf_counter
from typing import Sequence

import torch
from PIL import Image

from sglang.srt.mem_cache.scene_graph_memory.schema import (
    content_fingerprint,
    image_sha256,
)


class PrefixMismatch(ValueError):
    """A stored prefix cannot be used with the prepared prompt exactly as supplied."""


@dataclass(frozen=True)
class ImageSpan:
    pixel_sha256: str
    processor_patch_sha256: str
    original_size: tuple[int, int]
    processor_size: tuple[int, int]
    grid_thw: tuple[int, int, int]
    start: int
    stop: int


@dataclass(frozen=True)
class PreparedInput:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    rope_deltas: torch.Tensor
    image_grid_thw: torch.Tensor | None
    image_spans: tuple[ImageSpan, ...]
    image_patches: tuple[torch.Tensor, ...]
    preprocessing_seconds: float
    processor_cache_hits: int

    @property
    def logical_length(self) -> int:
        return self.input_ids.shape[-1]

    @property
    def metadata(self) -> dict:
        """Stable prompt compatibility data for tensor-only native cache storage."""
        return {
            "input_ids": self.input_ids.tolist(),
            "attention_mask": self.attention_mask.tolist(),
            "position_ids": self.position_ids.tolist(),
            "rope_deltas": self.rope_deltas.tolist(),
            "image_grid_thw": self.image_grid_thw.tolist()
            if self.image_grid_thw is not None
            else None,
            "image_spans": [asdict(span) for span in self.image_spans],
        }

    @property
    def fingerprint(self) -> str:
        return content_fingerprint(self.metadata)


@dataclass
class CacheState:
    cache: object
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    image_spans: tuple[ImageSpan, ...]
    retained_indices: tuple[int, ...]
    owner: object
    next_logits: torch.Tensor | None = None
    consumed: bool = False

    @property
    def logical_length(self) -> int:
        return self.input_ids.shape[-1]


@dataclass(frozen=True)
class GenerationResult:
    text: str
    generated_ids: tuple[int, ...]
    state: CacheState
    metrics: dict
    logits: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class _ProcessedImage:
    grid: tuple[int, int, int]
    patches: torch.Tensor
    patch_hash: str


class MiniO3Runtime:
    """Single unpadded conversation on a Qwen2.5-VL decoder in eval mode.

    `prefill`, `generate` and `truncate` consume any cache passed to them after
    validating it. Fork or compact before branching. Canonical prompt tensors
    stay on CPU. The only multimodal data sent to the model is the unprocessed
    suffix's image data, including when continuing after prior observations.
    """

    def __init__(
        self, model, processor, *, device, min_pixels=40_000, max_pixels=2_000_000
    ):
        if (
            getattr(type(model.config), "model_type", None) != "qwen2_5_vl"
            or model.training
        ):
            raise ValueError("The terminal runtime requires Qwen2.5-VL in eval mode.")
        if any(
            kind != "full_attention" for kind in model.config.text_config.layer_types
        ):
            raise ValueError(
                "The terminal runtime requires full-attention decoder layers."
            )
        if not 0 < min_pixels <= max_pixels:
            raise ValueError("Invalid visual processor pixel bounds.")
        self.model = model
        self.processor = processor
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.merge = model.config.vision_config.spatial_merge_size
        self.patch_size = model.config.vision_config.patch_size
        if self.patch_size != self.processor.image_processor.patch_size:
            raise ValueError("Vision patch size differs between processor and model.")
        if self.merge != self.processor.image_processor.merge_size:
            raise ValueError("Vision merge size differs between processor and model.")
        self.image_token = processor.image_token
        self.image_token_id = model.config.image_token_id
        self.owner = object()
        self._processed_images: OrderedDict[str, _ProcessedImage] = OrderedDict()

    def _processed_image(
        self, image: Image.Image, image_hash: str
    ) -> tuple[_ProcessedImage, bool]:
        entry = self._processed_images.get(image_hash)
        if entry is not None:
            self._processed_images.move_to_end(image_hash)
            return entry, True
        batch = self.processor.image_processor(
            images=[image],
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            return_tensors="pt",
        )
        raw_grid = batch["image_grid_thw"]
        patches = batch["pixel_values"].contiguous()
        if tuple(raw_grid.shape) != (1, 3):
            raise ValueError("Expected a single image grid.")
        grid = tuple(int(n) for n in raw_grid[0])
        t, h, w = grid
        if (
            t != 1
            or h < 1
            or w < 1
            or h % self.merge
            or w % self.merge
            or patches.shape[0] != prod(grid)
        ):
            raise ValueError(
                "Image grid and flattened patch rows disagree with Qwen spatial merge."
            )
        entry = _ProcessedImage(
            grid, patches, hashlib.sha256(patches.view(torch.uint8).numpy()).hexdigest()
        )
        self._processed_images[image_hash] = entry
        if len(self._processed_images) > 16:
            self._processed_images.popitem(last=False)
        return entry, False

    def _tokenize(self, messages, counts, assistant_token_ids, add_generation_prompt):
        override = assistant_token_ids or {}
        render = list(messages)
        markers = []
        for index, ids in sorted(override.items()):
            if (
                type(index) is not int
                or not 0 <= index < len(render)
                or render[index].get("role") != "assistant"
            ):
                raise ValueError(
                    "Raw assistant token IDs must correspond to an existing assistant message."
                )
            ids = tuple(ids)
            if not ids or any(
                type(token) is not int
                or not 0 <= token < self.model.config.text_config.vocab_size
                for token in ids
            ):
                raise ValueError(
                    "Expected nonempty raw assistant token IDs in vocabulary range."
                )
            marker = f"{{{{SceneCacheNativeAssistant{index}Boundary}}}}"
            render[index] = {**render[index], "content": marker}
            markers.append((marker, ids))
        prompt = self.processor.apply_chat_template(
            render, tokenize=False, add_generation_prompt=add_generation_prompt
        )
        chunks = prompt.split(self.image_token)
        if len(chunks) != 1 + len(counts):
            raise ValueError(
                "Number of image placeholder spans disagrees with observation count."
            )
        prompt = chunks[0] + "".join(
            self.image_token * count + part for count, part in zip(counts, chunks[1:])
        )
        if not markers:
            text = self.processor(text=[prompt], return_tensors="pt")
            return text["input_ids"], text["attention_mask"]

        tokenizer = self.processor.tokenizer
        eos_text = "<|im_end|>"
        eos_id = tokenizer.convert_tokens_to_ids(eos_text)
        combined = []
        for marker, ids in markers:
            if prompt.count(marker) != 1:
                raise ValueError(
                    "Assistant message boundary is ambiguous in rendered template."
                )
            before, prompt = prompt.split(marker, maxsplit=1)
            combined.extend(tokenizer(before, add_special_tokens=False)["input_ids"])
            combined.extend(ids)
            if ids[-1] == eos_id and prompt.startswith(eos_text):
                prompt = prompt[len(eos_text) :]
        combined.extend(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        ids = torch.tensor([combined], dtype=torch.long)
        return ids, torch.ones_like(ids)

    @torch.inference_mode()
    def prepare(
        self,
        messages,
        observations: Sequence[Image.Image],
        *,
        assistant_token_ids=None,
        add_generation_prompt=True,
        assistant_prefix="",
    ) -> PreparedInput:
        started = perf_counter()
        images = [
            image if image.mode == "RGB" else image.convert("RGB")
            for image in observations
        ]
        image_hashes = [image_sha256(image) for image in images]
        prepared_images = [
            self._processed_image(image, key)
            for image, key in zip(images, image_hashes)
        ]
        data = [entry for entry, _ in prepared_images]
        counts = [prod(entry.grid) // self.merge ** 2 for entry in data]
        ids, mask = self._tokenize(
            messages, counts, assistant_token_ids, add_generation_prompt
        )
        if assistant_prefix:
            if not add_generation_prompt:
                raise ValueError("An assistant prefix needs an open generation turn.")
            prefix_ids = self.processor.tokenizer(
                assistant_prefix, add_special_tokens=False, return_tensors="pt",
            ).input_ids
            ids = torch.cat((ids, prefix_ids), dim=-1)
            mask = torch.ones_like(ids)
        if (
            ids.ndim != 2
            or ids.shape[0] != 1
            or ids.shape[-1] < 1
            or not torch.all(mask == 1)
        ):
            raise ValueError("Expected a single unpadded, nonempty chat prompt.")
        rows = (ids[0] == self.image_token_id).nonzero().flatten().tolist()
        if len(rows) != sum(counts):
            raise ValueError("Actual image tokens disagree with the source grids.")
        tokens = ids[0].tolist()
        offset = 0
        spans = []
        for image, image_hash, entry, count in zip(images, image_hashes, data, counts):
            start = rows[offset]
            stop = start + count
            if (
                rows[offset : offset + count] != list(range(start, stop))
                or start == 0
                or stop == len(tokens)
                or tokens[start - 1] != self.model.config.vision_start_token_id
                or tokens[stop] != self.model.config.vision_end_token_id
            ):
                raise ValueError(
                    "Source grid does not align to a complete contiguous image placeholder span."
                )
            spans.append(
                ImageSpan(
                    image_hash, entry.patch_hash, image.size,
                    (entry.grid[2] * self.patch_size, entry.grid[1] * self.patch_size),
                    entry.grid, start, stop,
                )
            )
            offset += count
        grids = (
            torch.tensor([entry.grid for entry in data], dtype=torch.long)
            if data
            else None
        )
        positions, rope_deltas = self.model.model.get_rope_index(
            ids, image_grid_thw=grids, attention_mask=mask
        )
        if positions.shape != (3, 1, ids.shape[-1]):
            raise ValueError("Invalid native Qwen multimodal positions.")
        return PreparedInput(
            ids,
            mask,
            positions,
            rope_deltas,
            grids,
            tuple(spans),
            tuple(entry.patches for entry in data),
            perf_counter() - started,
            sum(hit for _, hit in prepared_images),
        )

    def _require_state(self, state: CacheState) -> None:
        if state.owner is not self.owner or state.consumed:
            raise PrefixMismatch(
                "Cache state belongs to a different runtime or has already been consumed."
            )
        if state.cache.get_seq_length() != len(state.retained_indices):
            raise PrefixMismatch(
                "Physical cache length disagrees with retained source indices."
            )

    @staticmethod
    def verify_prepared_prefix(
        short: PreparedInput | CacheState, long: PreparedInput
    ) -> int:
        """Verify the complete shorter logical prompt and return its length."""
        n = short.logical_length
        if n > long.logical_length:
            raise PrefixMismatch(
                "Prepared prompt is shorter than the cached logical prefix."
            )
        for name in ("input_ids", "attention_mask", "position_ids"):
            if not torch.equal(getattr(short, name), getattr(long, name)[..., :n]):
                raise PrefixMismatch(f"Cached prefix differs in {name}.")
        spans = tuple(span for span in long.image_spans if span.start < n)
        if spans != short.image_spans or any(span.stop > n for span in spans):
            raise PrefixMismatch(
                "Cached image identity, processor patches, grid or image span differs."
            )
        return n

    def verify_prefix(self, prepared: PreparedInput, state: CacheState) -> int:
        """Validate cache ownership and logical prefix without consuming it."""
        self._require_state(state)
        return self.verify_prepared_prefix(state, prepared)

    @torch.inference_mode()
    def fork(self, state: CacheState) -> CacheState:
        from sglang.srt.mem_cache.scene_graph_memory.terminal_kv import copy_cache

        self._require_state(state)
        return replace(state, cache=copy_cache(state.cache))

    @torch.inference_mode()
    def branch(self, state: CacheState) -> CacheState:
        from sglang.srt.mem_cache.scene_graph_memory.terminal_kv import branch_cache

        self._require_state(state)
        return replace(state, cache=branch_cache(state.cache))

    @torch.inference_mode()
    def compact(self, state: CacheState, logical_indices: Sequence[int]) -> CacheState:
        from sglang.srt.mem_cache.scene_graph_memory.terminal_kv import compact_cache

        self._require_state(state)
        indices = tuple(logical_indices)
        physical = {source: row for row, source in enumerate(state.retained_indices)}
        if (
            not indices
            or indices != tuple(sorted(set(indices)))
            or any(index not in physical for index in indices)
        ):
            raise ValueError(
                "Compaction requires increasing, unique retained logical indices."
            )
        cache = compact_cache(state.cache, [physical[index] for index in indices])
        return replace(state, cache=cache, retained_indices=indices, next_logits=None)

    @torch.inference_mode()
    def truncate(self, state: CacheState, logical_length: int) -> CacheState:
        self._require_state(state)
        if not 0 < logical_length < state.logical_length:
            raise ValueError(
                "Truncation must keep a strictly shorter, nonempty logical prefix."
            )
        if any(span.start <= logical_length <= span.stop for span in state.image_spans):
            raise ValueError(
                "Truncation must keep the entire image span and its end marker or remove it."
            )
        physical = bisect_left(state.retained_indices, logical_length)
        if not physical:
            raise ValueError("Truncation removed all physical cache rows.")
        state.consumed = True
        state.cache.crop(physical)
        return replace(
            state,
            input_ids=state.input_ids[..., :logical_length],
            attention_mask=state.attention_mask[..., :logical_length],
            position_ids=state.position_ids[..., :logical_length],
            image_spans=tuple(
                span for span in state.image_spans if span.stop < logical_length
            ),
            retained_indices=state.retained_indices[:physical],
            next_logits=None,
            consumed=False,
        )

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _forward(self, ids, positions, past, images=None):
        physical_length = past.get_seq_length() if past is not None else 0
        length = ids.shape[-1]
        output = self.model(
            input_ids=ids,
            attention_mask=torch.ones(
                (1, physical_length + length), dtype=torch.long, device=self.device
            ),
            position_ids=positions,
            cache_position=torch.arange(
                physical_length, physical_length + length, device=self.device
            ),
            past_key_values=past,
            use_cache=True,
            logits_to_keep=1,
            return_dict=True,
            **(images or {}),
        )
        cache = output.past_key_values
        if cache is None or cache.get_seq_length() != physical_length + length:
            raise RuntimeError("Native forward returned an incomplete cache.")
        return cache, output.logits[:, -1, :].float()

    def _prefill(self, prepared, state):
        reused = self.verify_prefix(prepared, state) if state is not None else 0
        retained = state.retained_indices if state is not None else ()
        active = [
            index
            for index, span in enumerate(prepared.image_spans)
            if span.start >= reused
        ]
        suffix = prepared.logical_length - reused
        metrics = {
            "logical_prompt_tokens": prepared.logical_length,
            "prefix_logical_tokens": reused,
            "prefix_physical_tokens": len(retained),
            "suffix_tokens": suffix,
            "vision_input_count": len(active),
            "vision_input_tokens": sum(
                prepared.image_spans[i].stop - prepared.image_spans[i].start
                for i in active
            ),
            "vision_patch_rows": sum(
                prepared.image_patches[i].shape[0] for i in active
            ),
            "cpu_processor_cache_bytes": sum(
                entry.patches.numel() * entry.patches.element_size()
                for entry in self._processed_images.values()
            ),
            "preprocessing_seconds": prepared.preprocessing_seconds,
            "processor_cache_hits": prepared.processor_cache_hits,
        }
        if not suffix:
            if state is None or state.next_logits is None:
                raise PrefixMismatch("Complete prefix has no valid next-token logits.")
            state.consumed = True
            return replace(state, consumed=False), {**metrics, "prefill_seconds": 0.0}
        ids = prepared.input_ids[:, reused:].to(self.device)
        positions = prepared.position_ids[..., reused:].to(self.device)
        images = {}
        if active:
            images = {
                "pixel_values": torch.cat(
                    [prepared.image_patches[i] for i in active]
                ).to(self.device),
                "image_grid_thw": prepared.image_grid_thw[active].to(self.device),
            }
        self._sync()
        started = perf_counter()
        if state is not None:
            state.consumed = True
        cache, logits = self._forward(
            ids, positions, state.cache if state is not None else None, images
        )
        self._sync()
        elapsed = perf_counter() - started
        complete = CacheState(
            cache,
            prepared.input_ids.clone(),
            prepared.attention_mask.clone(),
            prepared.position_ids.clone(),
            prepared.image_spans,
            retained + tuple(range(reused, prepared.logical_length)),
            self.owner,
            next_logits=logits,
        )
        return complete, {**metrics, "prefill_seconds": elapsed}

    @torch.inference_mode()
    def prefill(
        self, prepared: PreparedInput, *, state: CacheState | None = None
    ) -> CacheState:
        """Cache a complete prompt without sampling or decoding."""
        complete, _ = self._prefill(prepared, state)
        return complete

    @torch.inference_mode()
    def restore(
        self,
        prepared: PreparedInput,
        cache,
        *,
        retained_indices: Sequence[int] | None = None,
    ) -> CacheState:
        """Attach a loaded cache after its saved compatibility metadata was verified."""
        from sglang.srt.mem_cache.scene_graph_memory.terminal_kv import cache_stats

        indices = (
            tuple(range(prepared.logical_length))
            if retained_indices is None
            else tuple(retained_indices)
        )
        if (
            not indices
            or any(type(index) is not int for index in indices)
            or indices != tuple(sorted(set(indices)))
            or not 0 <= indices[0] <= indices[-1] < prepared.logical_length
        ):
            raise ValueError(
                "Loaded retained indices must be unique logical rows in source order."
            )
        stats = cache_stats(cache)
        text = self.model.config.text_config
        shape = [
            1,
            text.num_key_value_heads,
            len(indices),
            text.hidden_size // text.num_attention_heads,
        ]
        if (
            stats["sequence_length"] != len(indices)
            or stats["layer_count"] != text.num_hidden_layers
            or any(
                info["shape"] != shape
                or info["dtype"] != str(self.model.dtype)
                or torch.device(info["device"]) != self.device
                for layer in stats["layers"]
                for info in (layer["key"], layer["value"])
            )
        ):
            raise ValueError(
                "Loaded native cache shape, dtype or device differs from this decoder."
            )
        return CacheState(
            cache,
            prepared.input_ids.clone(),
            prepared.attention_mask.clone(),
            prepared.position_ids.clone(),
            prepared.image_spans,
            indices,
            self.owner,
        )

    @torch.inference_mode()
    def teacher_forced(
        self,
        prepared: PreparedInput,
        *,
        targets: Sequence[int],
        state: CacheState | None = None,
    ) -> GenerationResult:
        """Record one raw next-token logit vector per fixed token on the same trajectory."""
        targets = tuple(targets)
        if not targets or any(
            type(token) is not int
            or not 0 <= token < self.model.config.text_config.vocab_size
            for token in targets
        ):
            raise ValueError(
                "Teacher forcing requires a nonempty sequence of token IDs in vocabulary range."
            )
        return self.generate(
            prepared,
            state=state,
            max_new_tokens=len(targets),
            capture_logits=True,
            force_tokens=targets,
        )

    @torch.inference_mode()
    def generate(
        self,
        prepared: PreparedInput,
        *,
        state: CacheState | None = None,
        max_new_tokens: int,
        temperature: float = 0.0,
        generator: torch.Generator | None = None,
        stop_strings: Sequence[str] = ("</grounding>", "</answer>"),
        capture_logits: bool = False,
        force_tokens: Sequence[int] | None = None,
    ) -> GenerationResult:
        """Sample using native HF stopping/logit processors and an explicit RNG if supplied."""
        if max_new_tokens < 1 or not isfinite(temperature) or temperature < 0:
            raise ValueError(
                "Expected a positive token budget and a finite nonnegative temperature."
            )
        self._sync()
        started = perf_counter()
        n = prepared.logical_length
        config = deepcopy(self.model.generation_config)
        config.max_new_tokens = max_new_tokens
        config.max_length = n + max_new_tokens
        config.do_sample = temperature > 0
        if config.do_sample:
            config.temperature = temperature
        config.stop_strings = list(stop_strings) if stop_strings else None
        config.use_cache = True
        self.model._prepare_special_tokens(
            config, kwargs_has_attention_mask=True, device=self.device
        )
        if force_tokens is None:
            criteria = self.model._get_stopping_criteria(
                config,
                stopping_criteria=[],
                tokenizer=self.processor.tokenizer,
            )
        elif len(force_tokens) != max_new_tokens:
            raise ValueError(
                "Teacher-forced tokens must match the generation token count."
            )
        history = prepared.input_ids.to(self.device)
        processors = self.model._get_logits_processor(
            config,
            input_ids_seq_length=n,
            encoder_input_ids=history,
            device=self.device,
        )
        complete, metrics = self._prefill(prepared, state)
        cache, logits = complete.cache, complete.next_logits
        next_position = prepared.position_ids[..., -1:].to(self.device) + 1
        raw_logits = []
        generated = []
        stop_reason = "length"
        first_token_seconds = None
        for index in range(max_new_tokens):
            if capture_logits:
                raw_logits.append(logits.detach().cpu())
            if force_tokens is not None:
                token = torch.tensor([[force_tokens[index]]], device=self.device)
            else:
                scores = processors(history, logits.clone())
                if config.do_sample:
                    token = torch.multinomial(
                        scores.softmax(dim=-1), 1, generator=generator
                    )
                else:
                    token = scores.argmax(dim=-1, keepdim=True)
            history = torch.cat((history, token), dim=-1)
            generated.append(int(token.item()))
            if index == 0:
                first_token_seconds = perf_counter() - started
            if index + 1 == max_new_tokens:
                break
            if force_tokens is None and bool(criteria(history, None).item()):
                stop_reason = "stop"
                break
            cache, logits = self._forward(token, next_position + index, cache)
        self._sync()
        elapsed = perf_counter() - started
        cached_tokens = len(generated) - 1
        tail_positions = prepared.position_ids[..., -1:] + torch.arange(
            1, cached_tokens + 1
        )
        result_state = replace(
            complete,
            cache=cache,
            input_ids=torch.cat(
                (prepared.input_ids, torch.tensor([generated[:-1]], dtype=torch.long)),
                dim=-1,
            ),
            attention_mask=torch.ones((1, n + cached_tokens), dtype=torch.long),
            position_ids=torch.cat((prepared.position_ids, tail_positions), dim=-1),
            retained_indices=complete.retained_indices
            + tuple(range(n, n + cached_tokens)),
            next_logits=logits,
        )
        self._require_state(result_state)
        text = self.processor.decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return GenerationResult(
            text,
            tuple(generated),
            result_state,
            {
                **metrics,
                "generation_seconds": elapsed,
                "first_token_seconds": first_token_seconds,
                "after_first_token_seconds": elapsed - first_token_seconds,
                "generated_tokens": len(generated),
                "logical_cache_tokens": result_state.logical_length,
                "physical_cache_tokens": len(result_state.retained_indices),
                "stop_reason": stop_reason,
                "captured_logits": capture_logits,
            },
            tuple(raw_logits),
        )

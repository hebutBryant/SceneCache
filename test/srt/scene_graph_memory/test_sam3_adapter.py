"""CPU checks for imported masks and source-image coordinates."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageOps

from sglang.srt.mem_cache.scene_graph_memory.sam3_adapter import create_masks
from sglang.srt.mem_cache.scene_graph_memory.schema import BANK_REF, SCHEMA_VERSION, image_sha256, mask_sha256


class MaskImportTest(unittest.TestCase):
    def test_npz_empty_filter_and_oriented_dimensions(self):
        original = Image.new("RGB", (5, 3), "white")
        original.getexif()[274] = 6
        oriented = ImageOps.exif_transpose(original)
        self.assertEqual(oriented.size, (3, 5))
        stack = np.zeros((3, 5, 3), dtype=np.uint8)
        stack[1, 1, 2] = 255
        stack[2, 1:3, 1:3] = 1
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "masks.npz"
            np.savez(path, masks=stack)
            masks, provenance = create_masks(original, masks_file=path)
            self.assertEqual(provenance["backend"], "imported_masks")
            self.assertEqual(provenance["pixel_sha256"], hashlib.sha256(oriented.tobytes()).hexdigest())
            self.assertEqual(provenance["filtered"], [{"index": 0, "area": 0, "reason": "empty_mask"}])
            self.assertEqual(len(masks), 2)
            self.assertEqual(masks[0].mask.dtype, np.bool_)
            self.assertTrue(masks[0].mask[1, 2] and masks[1].mask[1, 2])
            masks, provenance = create_masks(original, masks_file=path, sam3_min_area=2)
            self.assertEqual(len(masks), 1)
            self.assertEqual(provenance["filtered"][1]["reason"], "below_min_area")

    def test_scene2graph_json_reference_and_pixel_mismatch(self):
        image = Image.new("RGB", (6, 4), "white")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            png = root / "binary.png"
            values = np.zeros((4, 6), dtype=np.uint8)
            values[0, 1] = 255
            Image.fromarray(values).save(png)
            manifest = {
                "tool": "sam",
                "image_size_wh": list(image.size),
                "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                "data": {"regions": [{"mask": png.name, "label": "object", "score": 0.9}]},
            }
            path = root / "result.json"
            path.write_text(json.dumps(manifest))
            masks, provenance = create_masks(image, masks_file=path)
            self.assertEqual(provenance["source"], {"tool": "sam"})
            self.assertEqual(masks[0].label, "object")
            self.assertEqual(masks[0].score, 0.9)
            self.assertEqual(masks[0].label_source, "imported_metadata")
            with self.assertRaisesRegex(ValueError, "different image pixels"):
                create_masks(Image.new("RGB", image.size, "black"), masks_file=path)

    def test_mask_shape_must_match(self):
        image = Image.new("RGB", (8, 4))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.npz"
            np.savez(path, masks=np.zeros((1, 8, 4), dtype=np.bool_))
            with self.assertRaisesRegex(ValueError, "oriented image coordinates"):
                create_masks(image, masks_file=path)

    def test_memory_manifest_import_and_payload_integrity(self):
        image = Image.new("RGB", (6, 4), "white")
        mask = np.zeros((4, 6), dtype=np.bool_)
        mask[1:3, 2:5] = True
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "masks").mkdir()
            reference = "masks/r0001.png"
            mask_path = root / reference
            Image.fromarray(mask).save(mask_path)
            payload = mask_path.read_bytes()
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "source_bank": {"id": BANK_REF, "file": "source_kv.safetensors"},
                "image_hash": image_sha256(image),
                "geometry": {"original_size": list(image.size)},
                "fingerprint": "sample-memory-fingerprint",
                "segmentation": {"backend": "sam3", "prompts": ["coat"]},
                "regions": [{
                    "region_id": "r0001", "mask_ref": reference, "mask_sha256": mask_sha256(mask),
                    "label": "coat", "label_source": "sam3_text_prompt", "score": 0.8,
                    "metadata": {"candidate_id": "Q001_C001"},
                }],
                "files": {reference: {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}},
            }
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest))
            masks, provenance = create_masks(image, masks_file=path)
            np.testing.assert_array_equal(masks[0].mask, mask)
            self.assertEqual(masks[0].label, "coat")
            self.assertEqual(masks[0].label_source, "sam3_text_prompt")
            self.assertEqual(masks[0].score, 0.8)
            self.assertEqual(masks[0].metadata["candidate_id"], "Q001_C001")
            self.assertEqual(provenance["source"]["segmentation"], manifest["segmentation"])
            with self.assertRaisesRegex(ValueError, "different image pixels"):
                create_masks(Image.new("RGB", image.size, "black"), masks_file=path)
            mask_path.write_bytes(payload + b"unexpected byte")
            with self.assertRaisesRegex(ValueError, "mask payload hash or size mismatch"):
                create_masks(image, masks_file=path)


if __name__ == "__main__":
    unittest.main()

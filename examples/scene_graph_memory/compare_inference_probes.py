"""Compare captured tensors for one selection and identical model inputs on CPU.

The tolerances are reporting thresholds. A fused BF16 attention forward can
introduce small differences which propagate to later layers. This comparison
does not establish an accuracy or end-to-end latency result.
"""

import argparse
import json
import math
from pathlib import Path

import torch


REQUIRED_LAYER_FIELDS = ("key", "value", "attention", "layer_output")
OPTIONAL_LAYER_FIELDS = ("raw_key", "raw_value")


def load_probe(path):
    probe = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(probe, dict) or not isinstance(probe.get("layers"), dict):
        raise ValueError(f"{path}: expected a dictionary with a layers mapping")
    layers = {}
    for key, values in probe["layers"].items():
        layer = int(key)
        if layer in layers or not isinstance(values, dict):
            raise ValueError(f"{path}: invalid captured layer {key!r}")
        layers[layer] = values
    if not layers:
        raise ValueError(f"{path}: expected at least one captured layer")
    return {"layers": layers, "logits": probe.get("logits")}


def compare_tensor(reference, candidate, atol, rtol, *, optional=False, logits=False):
    if reference is None or candidate is None:
        return {
            "status": "omitted" if optional else "missing",
            "reference_present": reference is not None,
            "candidate_present": candidate is not None,
        }
    if not isinstance(reference, torch.Tensor) or not isinstance(candidate, torch.Tensor):
        return {"status": "invalid_tensor"}
    result = {
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
    }
    if reference.shape != candidate.shape:
        return {"status": "shape_mismatch", **result}
    if reference.dtype == candidate.dtype:
        left = reference.contiguous().reshape(-1).view(torch.uint8)
        right = candidate.contiguous().reshape(-1).view(torch.uint8)
        result["bit_equal"] = torch.equal(left, right)
    else:
        result["bit_equal"] = False
    # The captured BF16 values are exactly representable in FP32. Accumulate
    # norms in FP64 so their reductions remain informative even at large sizes.
    reference = reference.float()
    candidate = candidate.float()
    finite = torch.isfinite(reference) & torch.isfinite(candidate)
    count = reference.numel()
    nonfinite = count - finite.count_nonzero().item()
    difference = (candidate - reference).abs()
    out_of_tolerance = ((difference > atol + rtol * reference.abs()) | ~finite).sum().item()
    result.update(
        status="compared" if not nonfinite else "nonfinite_values",
        elements=count,
        nonfinite_elements=nonfinite,
        out_of_tolerance=out_of_tolerance,
        allclose=out_of_tolerance == 0,
    )
    if nonfinite:
        difference = difference[finite]
        norm_reference = reference[finite]
    else:
        norm_reference = reference
    if difference.numel():
        rmse = difference.square().mean(dtype=torch.float64).sqrt().item()
        reference_rms = norm_reference.square().mean(dtype=torch.float64).sqrt().item()
        result.update(
            max_abs=difference.max().item(),
            mean_abs=difference.mean(dtype=torch.float64).item(),
            rmse=rmse,
            reference_rms=reference_rms,
            normalized_rmse=rmse / reference_rms if reference_rms else (0.0 if not rmse else None),
        )
    else:
        result.update(max_abs=None, mean_abs=None, rmse=None, reference_rms=None, normalized_rmse=None)
    if logits and count and not nonfinite:
        reference_argmax = reference.argmax(dim=-1).reshape(-1)
        candidate_argmax = candidate.argmax(dim=-1).reshape(-1)
        matches = (reference_argmax == candidate_argmax).count_nonzero().item()
        result.update(
            argmax_match_rows=matches,
            argmax_total_rows=reference_argmax.numel(),
            argmax_all_equal=matches == reference_argmax.numel(),
            last_reference_argmax=reference_argmax[-1].item(),
            last_candidate_argmax=candidate_argmax[-1].item(),
        )
    return result


def compare_probe(reference, candidate, atol, rtol):
    layer_ids = sorted(reference["layers"].keys() | candidate["layers"].keys())
    layers = {}
    for layer in layer_ids:
        baseline = reference["layers"].get(layer, {})
        trial = candidate["layers"].get(layer, {})
        layers[layer] = {
            name: compare_tensor(baseline.get(name), trial.get(name), atol, rtol, optional=optional)
            for names, optional in ((REQUIRED_LAYER_FIELDS, False), (OPTIONAL_LAYER_FIELDS, True))
            for name in names
        }
    logits = compare_tensor(reference["logits"], candidate["logits"], atol, rtol, logits=True)
    compared = [logits, *(metric for layer in layers.values() for metric in layer.values())]
    within_tolerance = all(
        metric.get("allclose") for metric in compared if metric["status"] != "omitted"
    )
    return {
        "layers": layers,
        "logits": logits,
        "all_comparable_tensors_within_tolerance": within_tolerance,
        "all_logits_argmax_equal": logits.get("argmax_all_equal"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True, action="append")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--atol", type=float, default=0.012)
    parser.add_argument("--rtol", type=float, default=0.012)
    parser.add_argument("--strict", action="store_true", help="Exit 1 on tolerance or logits argmax disagreement")
    args = parser.parse_args()
    if any(not math.isfinite(tol) or tol < 0 for tol in (args.atol, args.rtol)):
        parser.error("Tolerances must be finite and nonnegative")
    reference = load_probe(args.reference)
    comparisons = []
    for path in args.candidate:
        candidate = load_probe(path)
        comparisons.append({"candidate": str(path), **compare_probe(reference, candidate, args.atol, args.rtol)})
    report = {
        "reference": str(args.reference),
        "tolerance": {"atol": args.atol, "rtol": args.rtol},
        "interpretation": (
            "This compares the captured tensors for identical model inputs and index selection. "
            "The normalized RMSE is RMSE divided by the reference RMS. Bit equality is reported "
            "separately from tolerances because later-layer differences can accumulate in BF16."
        ),
        "comparisons": comparisons,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        if args.output.resolve() in {path.resolve() for path in (args.reference, *args.candidate)}:
            parser.error("--output must be different from all input probes")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    if args.strict and any(
        not item["all_comparable_tensors_within_tolerance"] or not item["all_logits_argmax_equal"]
        for item in comparisons
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

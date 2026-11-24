"""Build and inspect a persisted single-image scene graph KV memory."""

import argparse
import json
from pathlib import Path


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Segment an image and capture its full source KV")
    build.add_argument("--image", required=True, type=Path)
    build.add_argument("--model-path", required=True, type=Path)
    build.add_argument("--output-dir", required=True, type=Path)
    build.add_argument("--device", default="cuda:0", help="Qwen CUDA device (default: cuda:0)")
    build.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    segmentation = build.add_mutually_exclusive_group(required=True)
    segmentation.add_argument("--sam-prompts-file", type=Path, help="SAM3 concept prompts in JSON or plain text")
    segmentation.add_argument("--masks-file", type=Path, help="Existing PNG mask manifest or NPZ mask stack")
    sam = build.add_argument_group("SAM3 concept segmentation")
    sam.add_argument("--sam3-checkpoint", type=Path, help="Local Hugging Face SAM3 model directory")
    sam.add_argument("--sam3-source", type=Path, help="Scene2Graph checkout containing visual_tools/sam.py")
    sam.add_argument("--sam3-python", type=Path, help="Interpreter from the existing SAM3 environment")
    sam.add_argument("--sam3-device", help="SAM3 device (defaults to --device)")
    sam.add_argument("--sam3-score-threshold", type=float, default=0.5)
    sam.add_argument("--sam3-mask-threshold", type=float, default=0.5)
    sam.add_argument("--sam3-max-candidates", type=int, help="Limit candidates per concept, after score filtering")
    sam.add_argument("--sam3-min-area", type=int, default=1, help="Minimum mask area in source pixels")
    build.add_argument("--mask-token-overlap-threshold", type=float, default=0.0, help="Minimum footprint coverage; zero keeps every positive intersection")
    build.add_argument("--contact-mode", choices=["pixel_cells", "overlap_only"], default="pixel_cells")
    build.add_argument("--max-image-tokens", type=int, default=2048, help="Maximum post-merge visual tokens in a single prefill")
    build.add_argument("--max-image-edge", type=int, help="Optionally resize the longest edge before SGLang preprocessing")
    build.add_argument("--context-length", type=int, default=4096)
    build.add_argument("--mem-fraction-static", type=float)
    build.add_argument("--verify-capture", action="store_true", help="Compare a cold capture run against the same input with capture off")

    inspect = subparsers.add_parser("inspect", help="Validate and summarize a memory on CPU")
    inspect.add_argument("--memory-dir", required=True, type=Path)

    export = subparsers.add_parser("export-region", help="Export one region's real decoder K/V as safetensors")
    export.add_argument("--memory-dir", required=True, type=Path)
    export.add_argument("--region-id", required=True)
    export.add_argument("--output", required=True, type=Path)

    visualize = subparsers.add_parser("visualize", help="Write an offline HTML mask, token, and contact graph viewer")
    visualize.add_argument("--memory-dir", required=True, type=Path)
    visualize.add_argument("--output", required=True, type=Path)

    retrieve = subparsers.add_parser("agentic-retrieve", help="Resolve a semantic selection against a persisted source bank")
    retrieve.add_argument("--memory-dir", required=True, type=Path)
    retrieve.add_argument("--semantic-graph", required=True, type=Path)
    retrieve.add_argument("--question", required=True)
    retrieve.add_argument("--operation", help="An ID from the graph's saved operations")
    retrieve.add_argument("--token-budget", required=True, type=int)
    retrieve.add_argument("--background-tokens", default=0, type=int)
    retrieve.add_argument("--max-hops", default=0, type=int)
    return parser


def _build(args, parser):
    if args.sam_prompts_file and not all([args.sam3_checkpoint, args.sam3_source, args.sam3_python]):
        parser.error("build with --sam-prompts-file needs --sam3-checkpoint, --sam3-source and --sam3-python")

    from .builder import BuildConfig, build_image_memory
    from .store import inspect_summary, save_memory

    config = BuildConfig(
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        sam3_checkpoint=args.sam3_checkpoint,
        sam_prompts_file=args.sam_prompts_file,
        masks_file=args.masks_file,
        sam3_python=args.sam3_python,
        sam3_source=args.sam3_source,
        sam3_device=args.sam3_device or args.device,
        sam3_score_threshold=args.sam3_score_threshold,
        sam3_mask_threshold=args.sam3_mask_threshold,
        sam3_max_candidates=args.sam3_max_candidates,
        sam3_min_area=args.sam3_min_area,
        mask_token_overlap_threshold=args.mask_token_overlap_threshold,
        contact_mode=args.contact_mode,
        max_image_tokens=args.max_image_tokens,
        max_image_edge=args.max_image_edge,
        context_length=args.context_length,
        mem_fraction_static=args.mem_fraction_static,
        verify_capture=args.verify_capture,
    )
    memory = build_image_memory(args.image, config)
    save_memory(memory, args.output_dir)
    return {"output_dir": str(args.output_dir.resolve()), **inspect_summary(memory)}


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "build":
        result = _build(args, parser)
    elif args.command == "agentic-retrieve":
        from .agentic_index import AgenticKVIndex

        graph = json.loads(args.semantic_graph.read_text(encoding="utf-8"))
        index = AgenticKVIndex(args.memory_dir, graph)
        result = index.retrieve(
            args.question, graph["plan"], operation=args.operation, token_budget=args.token_budget,
            max_hops=args.max_hops, background_tokens=args.background_tokens,
        ).to_dict()
    else:
        from .store import inspect_summary, load_memory

        memory = load_memory(args.memory_dir)
        if args.command == "inspect":
            result = inspect_summary(memory)
        elif args.command == "export-region":
            if args.output.suffix.lower() != ".safetensors":
                parser.error("export-region --output must end in .safetensors")
            memory.export_region(args.region_id, args.output)
            result = {"region_id": args.region_id, "output": str(args.output.resolve())}
        else:
            from .visualization import write_visualization

            if args.output.suffix.lower() != ".html":
                parser.error("visualize --output must end in .html")
            write_visualization(memory, args.output)
            result = {"output": str(args.output.resolve())}

    print(json.dumps(result, indent=2, ensure_ascii=False))

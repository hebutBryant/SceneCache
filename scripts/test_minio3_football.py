#!/usr/bin/env python3
"""Run a local Mini-o3 visual search test and save the model's tool trace."""

import argparse
import json
import math
import re
import runpy
import tempfile
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


ROOT = Path(__file__).resolve().parents[1]
MINIO3 = Path("/home/hdd3/yaoyf/Mini-o3")
FOOTBALL_IMAGE = Path("/home/lipz/test_football.jpg")
QUESTION = (
    "What names are printed in white directly above the numbers 2 and 6 on the "
    "backs of the red jerseys? Use the zoom tool to verify the lettering on both "
    "jerseys. Give the two names in number order, separated by a slash."
)
EXPECTED = "CAPRICE / S-WHITE"
VISION_MARKER = "<|vision_start|><|image_pad|><|vision_end|>"
GROUNDING = re.compile(r"<grounding>(.*?)</grounding>", re.DOTALL)
ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
SOURCE = re.compile(r"original_image|observation_[1-9]\d*")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MINIO3 / "Mini-o3-7B-v1")
    parser.add_argument("--code-dir", type=Path, default=MINIO3 / "code")
    parser.add_argument("--image", type=Path, default=FOOTBALL_IMAGE)
    parser.add_argument("--question", default=QUESTION)
    parser.add_argument("--expected-answer", help="Check a custom question against this answer.")
    parser.add_argument("--output-dir", type=Path, help="New directory for the trace and crops.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0, help="Set to 0 for greedy decoding.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_turns < 1 or args.max_new_tokens < 1:
        parser.error("turn and token limits must be positive")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error("temperature must be finite and nonnegative")
    return args


def read_grounding(raw):
    action = json.loads(raw)
    if not isinstance(action, dict):
        raise ValueError("The grounding payload must be a JSON object.")
    bbox = action.get("bbox_2d")
    if not isinstance(bbox, list) or len(bbox) != 4 or any(
        not (type(coord) is int or type(coord) is float and math.isfinite(coord))
        for coord in bbox
    ):
        raise ValueError("bbox_2d must contain four finite normalized coordinates.")
    source = action.get("source")
    if not isinstance(source, str) or not SOURCE.fullmatch(source):
        raise ValueError("source must be original_image or observation_i, with i starting at 1.")
    return action


def check_answer(answer, expected):
    # Ignore case and formatting punctuation for the jersey transcription.
    return re.sub(r"[\W_]", "", answer.casefold()) == re.sub(r"[\W_]", "", expected.casefold())


def save_trace(path, trace):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


@torch.inference_mode()
def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    code = args.code_dir.resolve()
    prompts = runpy.run_path(str(code / "verl/trainer/constants.py"))
    tools = runpy.run_path(str(code / "verl/workers/rollout/vllm_rollout/function_tools.py"))
    prepare_crop = tools["prepare_grounding_inputs_multi_turn"]
    crop_image = tools["crop_image"]
    expected = args.expected_answer
    if expected is None and args.question == QUESTION and args.image.resolve() == FOOTBALL_IMAGE:
        expected = EXPECTED

    with Image.open(args.image) as original:
        observations = [original.convert("RGB")]
    if args.output_dir:
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True)
    else:
        run_root = ROOT / ".runtime/minio3_football"
        run_root.mkdir(parents=True, exist_ok=True)
        output_dir = Path(tempfile.mkdtemp(prefix="run-", dir=run_root))
    trace_path = output_dir / "trace.json"
    trace = {
        "model": str(args.model.resolve()),
        "image": str(args.image.resolve()),
        "image_size": list(observations[0].size),
        "question": args.question,
        "expected_answer": expected,
        "seed": args.seed,
        "temperature": args.temperature,
        "max_turns": args.max_turns,
        "max_new_tokens": args.max_new_tokens,
        "turns": [],
        "answer": None,
        "passed": None,
        "status": "running",
    }
    save_trace(trace_path, trace)
    print(f"Trace: {trace_path}\nLoading {args.model} on {device}…", flush=True)

    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
    ).eval().to(device)
    trace["load_seconds"] = round(time.perf_counter() - started, 2)
    messages = [
        {"role": "system", "content": prompts["TOOL_CROP_SYSTEM_PROMPT"]},
        {"role": "user", "content": VISION_MARKER + "\n" + args.question},
    ]
    generation = {
        "max_new_tokens": args.max_new_tokens,
        "use_cache": True,
        "do_sample": args.temperature > 0,
        "stop_strings": ["</grounding>", "</answer>"],
        "tokenizer": processor.tokenizer,
    }
    if args.temperature > 0:
        generation["temperature"] = args.temperature

    print(f"Question: {args.question}", flush=True)
    started = time.perf_counter()
    for turn_number in range(1, args.max_turns + 1):
        turn_start = time.perf_counter()
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[prompt], images=observations, return_tensors="pt",
            images_kwargs={"min_pixels": 40_000, "max_pixels": 2_000_000},
        ).to(device)
        generated = model.generate(**inputs, **generation)[0, inputs.input_ids.shape[1] :]
        text = processor.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        del inputs, generated
        entry = {
            "turn": turn_number,
            "generated_text": text,
            "seconds": round(time.perf_counter() - turn_start, 2),
        }
        trace["turns"].append(entry)
        print(f"\n--- Turn {turn_number} ({entry['seconds']} s) ---\n{text}", flush=True)
        messages.append({"role": "assistant", "content": text})

        grounding = GROUNDING.search(text)
        answer = ANSWER.search(text)
        if grounding:
            if turn_number == args.max_turns:
                trace["status"] = "turn_limit"
                break
            try:
                action = read_grounding(grounding[1])
                entry["grounding"] = action
                _, (source_image, pixels) = prepare_crop(
                    [action], observations, [image.size for image in observations],
                    use_relative_coordinates=True,
                )
                crop = crop_image(source_image, pixels, [], resize=1)
            except (ValueError, AssertionError) as exc:
                entry["tool_error"] = str(exc)
                response = (
                    f"ERROR occurs during grounding. Error Information: {exc}.\n"
                    + prompts["ERROR_INFO_MULTI_TURN_PROMPT"]
                )
                print(response, flush=True)
            else:
                index = len(observations)
                filename = f"observation_{index:02d}.png"
                crop.save(output_dir / filename)
                observations.append(crop)
                entry["observation"] = {
                    "index": index,
                    "file": filename,
                    "bbox_pixels": pixels,
                    "source_size": list(source_image.size),
                    "crop_size": list(crop.size),
                }
                response = prompts["TOOL_CALL_CROP_MULTI_TRUN_PROMPT"].format(
                    action_turn=turn_number - 1, observation_turn=index
                )
                print(f"Crop: {action['source']} {pixels} → {output_dir / filename}", flush=True)
            messages.append({"role": "user", "content": response})
        elif answer and answer[1].strip():
            trace["answer"] = answer[1].strip()
            trace["status"] = "answered"
            break
        else:
            trace["status"] = "no_complete_tool_call_or_answer"
            break
        save_trace(trace_path, trace)

    trace["inference_seconds"] = round(time.perf_counter() - started, 2)
    trace["crop_count"] = len(observations) - 1
    if expected is not None:
        trace["passed"] = check_answer(trace["answer"] or "", expected)
    save_trace(trace_path, trace)
    print(f"\nAnswer: {trace['answer']}\nStatus: {trace['status']}\nTrace: {trace_path}", flush=True)
    if expected is not None:
        print(f"Expected: {expected}\nResult: {'PASS' if trace['passed'] else 'FAIL'}", flush=True)
    return 0 if trace["answer"] is not None and trace["passed"] is not False else 1


if __name__ == "__main__":
    raise SystemExit(main())

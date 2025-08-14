#!/usr/bin/env python3
# coding: utf-8

import os
import time
import random
import argparse
import logging
import json

from codegeex.benchmark.utils import read_translation_dataset
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_args(parser):
    group = parser.add_argument_group("translation")

    group.add_argument("--model-name-or-path", type=str, required=True)
    group.add_argument("--src-path", type=str, required=True)
    group.add_argument("--tgt-path", type=str, required=True)
    group.add_argument("--dataset", type=str, default="humaneval")
    group.add_argument("--language-src-type", type=str, default=None)
    group.add_argument("--language-tgt-type", type=str, default=None)
    group.add_argument("--samples-per-problem", type=int, default=1)
    group.add_argument("--batch-size", type=int, default=8)
    group.add_argument("--temperature", type=float, default=1.0)
    group.add_argument("--top-p", type=float, default=0.9)
    group.add_argument("--top-k", type=int, default=0)
    group.add_argument("--max-tokens", type=int, default=512)
    group.add_argument("--output-file", type=str, default="translations.jsonl")
    group.add_argument("--seed", type=int, default=42)
    return parser


def main():
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()

    random.seed(args.seed)

    entries = read_translation_dataset(
        args.src_path,
        args.tgt_path,
        lang_src=args.language_src_type,
        lang_tgt=args.language_tgt_type,
        dataset_type=args.dataset,
    )

    # Duplicate each entry samples_per_problem times and shuffle
    tasks = []
    for entry in entries.values():
        tasks.extend([entry] * args.samples_per_problem)
    random.shuffle(tasks)

    total = len(tasks)
    logger.info(f"Loaded {len(entries)} problems, total tasks: {total}")

    # Initialize vLLM (H200ならbf16推奨 / device指定は不要)
    llm = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.95,
        kv_cache_dtype="fp8",
        max_num_batched_tokens=16384,
        max_num_seqs=args.batch_size,
    )

    # 共通の SamplingParams
    base_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        stop=["<|endoftext|>", "</s>", "<|EOT|>", "<|im_end|>"]
    )

    start_time = time.perf_counter()
    n_done = 0

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    with open(args.output_file, "w", encoding="utf-8") as outfile:
        # バッチでまとめて生成
        for i in range(0, total, args.batch_size):
            batch = tasks[i : i + args.batch_size]

            # vLLMは「プロンプトのリスト」を受け取る
            prompts = []
            task_ids = []
            for entry in batch:
                prompt = entry.get("src", entry.get("prompt", ""))
                prompts.append(prompt)
                task_ids.append(entry["task_id"])

            # 生成（出力は入力順に整列）
            outputs = llm.generate(prompts, sampling_params=base_params)

            # 書き出し
            for j, out in enumerate(outputs):
                # out.outputs は候補のリスト（通常は1つ）:
                text = out.outputs[0].text if out.outputs else ""
                record = {
                    "task_id": task_ids[j],
                    "generated": text,
                }
                outfile.write(json.dumps(record, ensure_ascii=False) + "\n")

            n_done += len(batch)
            elapsed = time.perf_counter() - start_time
            logger.info(f"Processed {n_done}/{total} tasks (elapsed: {elapsed:.1f}s)")

    logger.info(f"All done. Results written to {args.output_file}")


if __name__ == "__main__":
    main()

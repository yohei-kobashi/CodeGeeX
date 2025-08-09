#!/usr/bin/env python3
# coding: utf-8

import os
import time
import random
import argparse
import logging
import json

from codegeex.benchmark.utils import read_translation_dataset
from vllm import LLM, SamplingParams, Request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_args(parser):
    """Argument definitions for vLLM-based code translation."""
    group = parser.add_argument_group("translation")

    group.add_argument(
        "--model-name-or-path",
        type=str,
        required=True,
        help="Path to a vLLM-compatible model or Hugging Face repo name",
    )
    group.add_argument(
        "--src-path",
        type=str,
        required=True,
        help="Path to the source code JSONL file",
    )
    group.add_argument(
        "--tgt-path",
        type=str,
        required=False,
        help="Path to the target code JSONL file (optional reference)",
    )
    group.add_argument(
        "--dataset",
        type=str,
        default="humaneval",
        help="Dataset type",
    )
    group.add_argument(
        "--language-src-type",
        type=str,
        default=None,
        help="Identifier for the source language",
    )
    group.add_argument(
        "--language-tgt-type",
        type=str,
        default=None,
        help="Identifier for the target language",
    )
    group.add_argument(
        "--samples-per-problem",
        type=int,
        default=1,
        help="Number of samples to generate per problem",
    )
    group.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for vLLM",
    )
    group.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )
    group.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling probability threshold",
    )
    group.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Top-k sampling token limit",
    )
    group.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate",
    )
    group.add_argument(
        "--output-file",
        type=str,
        default="translations.jsonl",
        help="Output JSONL file for generation results",
    )
    group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    return parser


def main():
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Load the dataset
    entries = read_translation_dataset(
        args.src_path,
        args.tgt_path or "",
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

    # Initialize vLLM
    llm = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=1,    # Assuming single GPU/worker
        dtype="fp16",               # Change dtype as needed
        device="cuda",
    )

    # Prepare output file
    outfile = open(args.output_file, "w", encoding="utf-8")

    start_time = time.perf_counter()
    # Process in batches sequentially
    for idx in range(0, total, args.batch_size):
        batch = tasks[idx : idx + args.batch_size]
        # Create list of requests
        requests = []
        for entry in batch:
            prompt = entry.get("src", entry.get("prompt", ""))
            req_id = entry["task_id"]
            sampling_params = SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_tokens=args.max_tokens,
            )
            requests.append(Request(prompt=prompt, request_id=req_id, sampling_params=sampling_params))

        # Run generation
        outputs = llm.generate(requests)

        # Write results
        for out in outputs:
            record = {
                "task_id": out.request_id,
                "generated": out.generated_text,
            }
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Progress log
        elapsed = time.perf_counter() - start_time
        logger.info(f"Processed {min(idx + args.batch_size, total)}/{total} tasks (elapsed: {elapsed:.1f}s)")

    outfile.close()
    logger.info(f"All done. Results written to {args.output_file}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# coding: utf-8

import os
import time
import random
import argparse
import logging
import json

from codegeex.benchmark.utils import read_dataset, process_extra_prompt
from vllm import LLM, SamplingParams, Request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_args(parser):
    """Argument definitions for vLLM-based HumanEval generation."""
    group = parser.add_argument_group("generation")

    group.add_argument(
        "--model-name-or-path",
        type=str,
        required=True,
        help="Path to a vLLM-compatible model or Hugging Face repo name",
    )
    group.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to the HumanEval JSONL file",
    )
    group.add_argument(
        "--dataset",
        type=str,
        default="humaneval",
        help="Dataset type (passed to read_dataset)",
    )
    group.add_argument(
        "--prompt-type",
        type=str,
        default="notag",
        help="Prompt formatting type",
    )
    group.add_argument(
        "--extra-prompt",
        type=str,
        default=None,
        help="Extra prompt prefix to use",
    )
    group.add_argument(
        "--language-type",
        type=str,
        default=None,
        help="Identifier for programming language processing",
    )
    group.add_argument(
        "--samples-per-problem",
        type=int,
        default=200,
        help="Number of samples to generate per problem",
    )
    group.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for vLLM inference",
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
        default=0.0,
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
        default=1024,
        help="Maximum number of tokens to generate",
    )
    group.add_argument(
        "--output-file",
        type=str,
        default="./output_humaneval.jsonl",
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

    # set random seed
    random.seed(args.seed)

    # load dataset
    entries = read_dataset(args.input_path, dataset_type=args.dataset)

    # process prompts (e.g., insert extra prompt or language tags)
    for entry in entries.values():
        entry["prompt"] = process_extra_prompt(entry["prompt"], args.language_type)
        if args.extra_prompt:
            entry["prompt"] = args.extra_prompt + entry["prompt"]

    # build task list: duplicate per samples_per_problem and shuffle
    tasks = []
    for entry in entries.values():
        tasks.extend([entry] * args.samples_per_problem)
    random.shuffle(tasks)

    total = len(tasks)
    logger.info(f"Loaded {len(entries)} problems, total tasks: {total}")

    # initialize vLLM
    llm = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=1,
        dtype="fp16",
        device="cuda",
    )

    # open output file
    with open(args.output_file, "w", encoding="utf-8") as outfile:
        start_time = time.perf_counter()

        # process in batches
        for idx in range(0, total, args.batch_size):
            batch = tasks[idx : idx + args.batch_size]
            requests = []
            for entry in batch:
                req_id = entry["task_id"]
                sampling_params = SamplingParams(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    max_tokens=args.max_tokens,
                )
                requests.append(Request(
                    prompt=entry["prompt"],
                    request_id=req_id,
                    sampling_params=sampling_params,
                ))

            # run generation
            outputs = llm.generate(requests)

            # write results
            for out in outputs:
                record = {
                    "task_id": out.request_id,
                    "generated": out.generated_text,
                }
                outfile.write(json.dumps(record, ensure_ascii=False) + "\n")

            # log progress
            elapsed = time.perf_counter() - start_time
            logger.info(f"Processed {min(idx + args.batch_size, total)}/{total} tasks (elapsed: {elapsed:.1f}s)")

    logger.info(f"All done. Results written to {args.output_file}")


if __name__ == "__main__":
    main()

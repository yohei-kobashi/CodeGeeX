#!/usr/bin/env python3
# coding: utf-8

import os
import time
import random
import argparse
import logging
import json
import urllib.request
import urllib.error

from codegeex.benchmark.utils import read_translation_dataset, cleanup_code

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
    group.add_argument("--batch-size", type=int, default=1)
    group.add_argument("--temperature", type=float, default=0.8)
    group.add_argument("--top-p", type=float, default=0.95)
    group.add_argument("--top-k", type=int, default=0)
    group.add_argument("--max-tokens", type=int, default=1024)
    group.add_argument("--output-file", type=str, default="translations.jsonl")
    group.add_argument("--seed", type=int, default=42)
    # When provided, use external vLLM OpenAI-compatible server instead of local vLLM
    group.add_argument(
        "--server-url",
        type=str,
        default=None,
        help="Base URL of vLLM OpenAI server (e.g., http://localhost:8000/v1). If set, use HTTP API instead of local vLLM.",
    )
    group.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY", None),
        help="Optional API key for the server (sent as Bearer token).",
    )
    group.add_argument(
        "--request-timeout",
        type=float,
        default=120.0,
        help="HTTP request timeout in seconds for server mode.",
    )
    return parser


def _patch_hf_autoconfig_register_allow_duplicates():
    """Work around duplicate AutoConfig.register keys (e.g., 'aimv2').

    Some environments ship Transformers versions that already register certain
    keys vLLM attempts to register. This patch makes AutoConfig.register
    tolerant by setting exist_ok=True and swallowing duplicate errors.
    """
    try:
        from transformers.models.auto.configuration_auto import AutoConfig
        orig_register = AutoConfig.register

        def safe_register(key, config, exist_ok=False):  # type: ignore
            try:
                return orig_register(key, config, exist_ok=True)
            except Exception:
                return None

        AutoConfig.register = safe_register  # type: ignore
        logger.info("Patched Transformers AutoConfig.register to allow duplicates.")
    except Exception as e:
        logger.warning(f"AutoConfig.register patch failed or not needed: {e}")


def _call_vllm_server(base_url,
                      model,
                      prompt,
                      *,
                      temperature,
                      top_p,
                      top_k,
                      max_tokens,
                      stop,
                      api_key,
                      timeout):
    """Call vLLM's OpenAI-compatible /completions endpoint for a single prompt.

    Returns generated text or empty string on failure.
    """
    url = base_url.rstrip("/") + "/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_tokens": max_tokens,
        "stop": stop,
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            obj = json.loads(body)
            # Expect OpenAI-style completions response
            choices = obj.get("choices", [])
            if choices:
                # Some servers may include leading/trailing spaces/newlines; keep as-is
                return choices[0].get("text", "") or ""
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8")
        except Exception:
            err_body = str(e)
        logging.error(f"HTTPError from server: {e.code} {e.reason} body={err_body}")
    except urllib.error.URLError as e:
        logging.error(f"URLError contacting server: {e}")
    except Exception as e:
        logging.exception(f"Unexpected error contacting server: {e}")
    return ""


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

    use_server = args.server_url is not None and len(args.server_url.strip()) > 0

    # Common stop markers to curb extra language blocks and markdown
    lang_stops = [
        "\nC++:", "\nJava:", "\nJavaScript:", "\nGo:", "\nPython:", "\nRust:",
        "```",
    ]
    if not use_server:
        # Delay-import vLLM only if using local inference
        _patch_hf_autoconfig_register_allow_duplicates()
        from vllm import LLM, SamplingParams
        # Initialize vLLM (H200ならbf16推奨 / device指定は不要)
        llm = LLM(
            model=args.model_name_or_path,
            tensor_parallel_size=1,
            dtype="bfloat16",
            gpu_memory_utilization=0.95,
            kv_cache_dtype="fp8",
            max_num_batched_tokens=16384,
            max_num_seqs=args.batch_size,
            enforce_eager=True,
        )

        # 共通の SamplingParams
        base_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
            stop=["<|endoftext|>", "</s>", "<|EOT|>", "<|im_end|>"] + lang_stops,
        )
    else:
        # In server mode, keep stop list as a simple Python list for HTTP payloads
        base_params = {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_tokens": args.max_tokens,
            "stop": ["<|endoftext|>", "</s>", "<|EOT|>", "<|im_end|>"] + lang_stops,
        }

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
                # Encourage code-only outputs for the target language
                if args.language_tgt_type:
                    tgt = args.language_tgt_type
                    prompt = (
                        f"{prompt}\n\nNote: Output only the {tgt} function implementation that matches the above declaration. "
                        f"Do not include explanations, markdown, or any other language sections."
                    )
                prompts.append(prompt)
                task_ids.append(entry["task_id"])

            if not use_server:
                # 生成（出力は入力順に整列）
                outputs = llm.generate(prompts, sampling_params=base_params)

                # 書き出し
                for j, out in enumerate(outputs):
                    # out.outputs は候補のリスト（通常は1つ）:
                    text = out.outputs[0].text if out.outputs else ""
                    # Post-process to keep target-language code only
                    cleaned = cleanup_code(
                        text,
                        language_type=(args.language_tgt_type or args.language_src_type or ""),
                        dataset=args.dataset,
                    )
                    record = {
                        "task_id": task_ids[j],
                        "prompt": prompts[j],
                        "generation": cleaned,
                        "generated": text,
                    }
                    outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
            else:
                # Server mode: call HTTP API for each prompt in the batch
                for j, prompt in enumerate(prompts):
                    text = _call_vllm_server(
                        args.server_url,
                        args.model_name_or_path,
                        prompt,
                        temperature=base_params["temperature"],
                        top_p=base_params["top_p"],
                        top_k=base_params["top_k"],
                        max_tokens=base_params["max_tokens"],
                        stop=base_params["stop"],
                        api_key=args.api_key,
                        timeout=args.request_timeout,
                    )
                    cleaned = cleanup_code(
                        text,
                        language_type=(args.language_tgt_type or args.language_src_type or ""),
                        dataset=args.dataset,
                    )
                    record = {
                        "task_id": task_ids[j],
                        "prompt": prompts[j],
                        "generation": cleaned,
                        "generated": text,
                    }
                    outfile.write(json.dumps(record, ensure_ascii=False) + "\n")

            n_done += len(batch)
            elapsed = time.perf_counter() - start_time
            logger.info(f"Processed {n_done}/{total} tasks (elapsed: {elapsed:.1f}s)")

    logger.info(f"All done. Results written to {args.output_file}")


if __name__ == "__main__":
    main()

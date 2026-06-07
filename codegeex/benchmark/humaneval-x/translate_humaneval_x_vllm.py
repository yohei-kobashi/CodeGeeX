#!/usr/bin/env python3
# coding: utf-8

import os
import time
import random
import argparse
import logging
import json
import subprocess
import urllib.error
import urllib.request

from codegeex.benchmark.utils import (cleanup_code,
                                      is_code_generation_finished,
                                      read_translation_dataset)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_args(parser):
    group = parser.add_argument_group("translation")

    group.add_argument("--model-name-or-path", type=str, required=True)
    group.add_argument(
        "--tokenizer-name-or-path",
        type=str,
        default=None,
        help="Optional tokenizer path/name. Useful when --model-name-or-path is a vLLM served-model alias.",
    )
    group.add_argument("--src-path", type=str, required=True)
    group.add_argument("--tgt-path", type=str, required=True)
    group.add_argument("--dataset", type=str, default="humaneval")
    group.add_argument("--language-src-type", type=str, default=None)
    group.add_argument("--language-tgt-type", type=str, default=None)
    group.add_argument("--samples-per-problem", type=int, default=20)
    group.add_argument("--batch-size", type=int, default=1)
    group.add_argument("--do-sample", type=lambda x: str(x).lower() in ("1", "true", "yes", "y"), default=True)
    group.add_argument("--temperature", type=float, default=0.8)
    group.add_argument("--top-p", type=float, default=0.95)
    group.add_argument("--top-k", type=int, default=0)
    group.add_argument("--min-p", type=float, default=0.0)
    group.add_argument("--presence-penalty", type=float, default=0.0)
    group.add_argument("--repetition-penalty", type=float, default=1.0)
    group.add_argument("--max-tokens", type=int, default=1024)
    group.add_argument("--output-file", type=str, default="translations.jsonl")
    group.add_argument("--seed", type=int, default=42)
    group.add_argument("--use-sft-prompt-template", type=bool, default=True)
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
        default=600.0,
        help="HTTP request timeout in seconds for server mode.",
    )
    group.add_argument(
        "--server-request-retries",
        type=int,
        default=3,
        help="Number of HTTP retries for each server-mode batch.",
    )
    group.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip generation when output-file already has the expected number of JSONL rows.",
    )
    return parser


def _truncate_to_finished(code: str, language: str, dataset: str) -> str:
    """Truncate the generated snippet at the first point where
    is_code_generation_finished would return True when streaming."""
    if not code or not language:
        return code

    language = language.lower()
    collected = []
    finished = None
    for line in code.splitlines(keepends=True):
        collected.append(line)
        candidate = "".join(collected)
        try:
            if is_code_generation_finished(candidate,
                                            language_type=language,
                                            dataset=dataset):
                finished = candidate
                break
        except Exception:
            break

    return finished if finished is not None else code


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


def _normalize_cuda_visible_devices_for_vllm():
    """Convert GPU UUID entries in CUDA_VISIBLE_DEVICES to numeric GPU indices.

    Some schedulers expose GPUs as UUIDs (for example
    CUDA_VISIBLE_DEVICES=GPU-...). CUDA itself accepts this, but some vLLM
    versions try to parse CUDA_VISIBLE_DEVICES entries as integers while
    importing CUDA quantization utilities. Normalize before importing vLLM.
    """
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible_devices or "GPU-" not in visible_devices:
        return

    requested = [device.strip() for device in visible_devices.split(",")]
    if not any(device.startswith("GPU-") for device in requested):
        return

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception as e:
        logger.warning(
            "CUDA_VISIBLE_DEVICES contains GPU UUIDs, but failed to query "
            f"nvidia-smi for numeric indices: {e}"
        )
        return

    uuid_to_index = {}
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 1)]
        if len(parts) == 2:
            index, uuid = parts
            uuid_to_index[uuid] = index

    normalized = []
    for device in requested:
        if device.startswith("GPU-"):
            index = uuid_to_index.get(device)
            if index is None:
                logger.warning(
                    "Could not map CUDA_VISIBLE_DEVICES UUID %s to a numeric "
                    "GPU index; leaving CUDA_VISIBLE_DEVICES unchanged.",
                    device,
                )
                return
            normalized.append(index)
        else:
            normalized.append(device)

    normalized_visible_devices = ",".join(normalized)
    os.environ["CUDA_VISIBLE_DEVICES"] = normalized_visible_devices
    logger.info(
        "Normalized CUDA_VISIBLE_DEVICES for vLLM: %s -> %s",
        visible_devices,
        normalized_visible_devices,
    )


def _call_vllm_server(base_url,
                      model,
                      prompt,
                      *,
                      temperature,
                      top_p,
                      top_k,
                      min_p,
                      presence_penalty,
                      repetition_penalty,
                      max_tokens,
                      stop,
                      api_key,
                      timeout):
    """Call vLLM's OpenAI-compatible /completions endpoint for a single prompt.

    Returns generation metadata. Text is empty on failure.
    """
    url = base_url.rstrip("/") + "/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "presence_penalty": presence_penalty,
        "repetition_penalty": repetition_penalty,
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
                choice = choices[0]
                usage = obj.get("usage", {}) or {}
                return {
                    "text": choice.get("text", "") or "",
                    "finish_reason": choice.get("finish_reason"),
                    "completion_tokens": usage.get("completion_tokens"),
                }
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
    return {
        "text": "",
        "finish_reason": None,
        "completion_tokens": None,
    }


def _call_vllm_server_batch(base_url,
                            model,
                            prompts,
                            *,
                            temperature,
                            top_p,
                            top_k,
                            min_p,
                            presence_penalty,
                            repetition_penalty,
                            max_tokens,
                            stop,
                            api_key,
                            timeout,
                            retries):
    """Call vLLM's OpenAI-compatible /completions endpoint for a prompt batch."""
    if not prompts:
        return []

    url = base_url.rstrip("/") + "/completions"
    payload = {
        "model": model,
        "prompt": prompts,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "presence_penalty": presence_penalty,
        "repetition_penalty": repetition_penalty,
        "max_tokens": max_tokens,
        "stop": stop,
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    empty_result = {
        "text": "",
        "finish_reason": None,
        "completion_tokens": None,
    }
    last_error = None
    retries = max(1, retries)
    for attempt in range(1, retries + 1):
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
                obj = json.loads(body)
                break
        except urllib.error.HTTPError as e:
            try:
                err_body = e.read().decode("utf-8")
            except Exception:
                err_body = str(e)
            last_error = f"HTTPError from server: {e.code} {e.reason} body={err_body}"
        except urllib.error.URLError as e:
            last_error = f"URLError contacting server: {e}"
        except Exception as e:
            last_error = f"Unexpected error contacting server: {e}"

        logger.warning(
            "Server batch request failed on attempt %d/%d: %s",
            attempt,
            retries,
            last_error,
        )
        if attempt < retries:
            time.sleep(min(30, 2 ** attempt))
    else:
        raise RuntimeError(f"Server batch request failed after {retries} attempts: {last_error}")

    choices = obj.get("choices", []) or []
    usage = obj.get("usage", {}) or {}
    completion_tokens = usage.get("completion_tokens")
    results = [dict(empty_result) for _ in prompts]

    for fallback_index, choice in enumerate(choices):
        index = choice.get("index", fallback_index)
        if not isinstance(index, int) or index < 0 or index >= len(results):
            logger.warning("Ignoring completion with invalid index: %s", index)
            continue
        results[index] = {
            "text": choice.get("text", "") or "",
            "finish_reason": choice.get("finish_reason"),
            "completion_tokens": completion_tokens,
        }

    if len(choices) != len(prompts):
        logger.warning(
            "Server returned %d choices for %d prompts.",
            len(choices),
            len(prompts),
        )

    return results


def _output_token_count(sequence, text, tokenizer=None):
    token_ids = getattr(sequence, "token_ids", None) if sequence is not None else None
    if token_ids is not None:
        return len(token_ids)
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            return None
    return None


def _reached_max_tokens(finish_reason, output_tokens, max_tokens):
    normalized_finish_reason = (finish_reason or "").lower()
    if normalized_finish_reason == "length":
        return True
    return output_tokens is not None and output_tokens >= max_tokens


def _jsonl_line_count(path):
    try:
        with open(path, "r", encoding="utf-8") as infile:
            return sum(1 for _ in infile)
    except FileNotFoundError:
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()

    random.seed(args.seed)

    use_server = args.server_url is not None and len(args.server_url.strip()) > 0
    if not use_server:
        _normalize_cuda_visible_devices_for_vllm()

    tokenizer = None
    tokenizer_name_or_path = args.tokenizer_name_or_path or args.model_name_or_path
    if tokenizer_name_or_path:
        from transformers import AutoTokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name_or_path,
                trust_remote_code=True,
            )
        except AttributeError as e:
            if "'list' object has no attribute 'keys'" in str(e):
                raise RuntimeError(
                    "Failed to load tokenizer because the model tokenizer config "
                    "appears to define extra_special_tokens as a list, while this "
                    "Transformers version expects a dict. Fix the model's "
                    "tokenizer_config.json or use a compatible Transformers "
                    f"version. tokenizer={tokenizer_name_or_path}"
                ) from e
            raise

    entries = read_translation_dataset(
        args.src_path,
        args.tgt_path,
        lang_src=args.language_src_type,
        lang_tgt=args.language_tgt_type,
        dataset_type=args.dataset,
        use_sft_prompt_template=args.use_sft_prompt_template,
        tokenizer=tokenizer
    )

    # Duplicate each entry samples_per_problem times and shuffle
    tasks = []
    for entry in entries.values():
        tasks.extend([entry] * args.samples_per_problem)
    random.shuffle(tasks)

    total = len(tasks)
    logger.info(f"Loaded {len(entries)} problems, total tasks: {total}")

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
        sampling_temperature = args.temperature if args.do_sample else 0.0
        base_params = SamplingParams(
            temperature=sampling_temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            presence_penalty=args.presence_penalty,
            repetition_penalty=args.repetition_penalty,
            max_tokens=args.max_tokens,
            stop=["<|endoftext|>", "</s>", "<|EOT|>", "<|im_end|>"] + lang_stops,
        )
    else:
        # In server mode, keep stop list as a simple Python list for HTTP payloads
        base_params = {
            "temperature": args.temperature if args.do_sample else 0.0,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "presence_penalty": args.presence_penalty,
            "repetition_penalty": args.repetition_penalty,
            "max_tokens": args.max_tokens,
            "stop": ["<|endoftext|>", "</s>", "<|EOT|>", "<|im_end|>"] + lang_stops,
        }

    start_time = time.perf_counter()
    n_done = 0
    max_token_generations = 0
    total_generations = 0

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if args.skip_existing:
        existing_lines = _jsonl_line_count(args.output_file)
        if existing_lines >= total:
            logger.info(
                "Skipping existing output with %d/%d rows: %s",
                existing_lines,
                total,
                args.output_file,
            )
            return

    tmp_output_file = args.output_file + ".tmp"

    with open(tmp_output_file, "w", encoding="utf-8") as outfile:
        # バッチでまとめて生成
        for i in range(0, total, args.batch_size):
            batch = tasks[i : i + args.batch_size]

            # vLLMは「プロンプトのリスト」を受け取る
            prompts = []
            task_ids = []
            for entry in batch:
                # Use prepared prompt (may be chat-template applied)
                prompt = entry.get("prompt", "")
                prompts.append(prompt)
                task_ids.append(entry["task_id"])

            if not use_server:
                # 生成（出力は入力順に整列）
                outputs = llm.generate(prompts, sampling_params=base_params)

                # 書き出し
                for j, out in enumerate(outputs):
                    # out.outputs は候補のリスト（通常は1つ）:
                    seq = out.outputs[0] if out.outputs else None
                    text = seq.text if seq else ""
                    lang = (args.language_tgt_type or args.language_src_type or "")
                    truncated = _truncate_to_finished(text, lang, args.dataset)
                    cleaned = cleanup_code(truncated, language_type=lang or "", dataset=args.dataset)

                    # Megatron-format metadata: cumulative logprob (if available),
                    # finishing status, and full token ids (prompt + generation).
                    if seq is not None:
                        try:
                            cumulative_logprob = float(seq.cumulative_logprob)
                        except Exception:
                            cumulative_logprob = 0.0
                        finish_reason = getattr(seq, "finish_reason", None)
                        finish = 2 if finish_reason in ("stop", "eos") else 1
                        try:
                            prompt_token_ids = list(getattr(out, "prompt_token_ids", []) or [])
                            generated_token_ids = list(seq.token_ids or [])
                            token_ids = prompt_token_ids + generated_token_ids
                        except Exception:
                            token_ids = []
                        output_tokens = _output_token_count(seq, text, tokenizer=tokenizer)
                    else:
                        cumulative_logprob = 0.0
                        finish_reason = None
                        finish = 1
                        token_ids = []
                        output_tokens = None

                    reached_max_tokens = _reached_max_tokens(
                        finish_reason,
                        output_tokens,
                        args.max_tokens,
                    )
                    total_generations += 1
                    if reached_max_tokens:
                        max_token_generations += 1

                    record = {
                        "task_id": task_ids[j],
                        "prompt": prompts[j],
                        "generation": cleaned,
                        "scores": cumulative_logprob,
                        "finish": finish,
                        "output": token_ids,
                        "finish_reason": finish_reason,
                        "output_tokens": output_tokens,
                        "reached_max_tokens": reached_max_tokens,
                    }
                    outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
            else:
                # Server mode: call HTTP API once per prompt batch.
                generation_infos = _call_vllm_server_batch(
                    args.server_url,
                    args.model_name_or_path,
                    prompts,
                    temperature=base_params["temperature"],
                    top_p=base_params["top_p"],
                    top_k=base_params["top_k"],
                    min_p=base_params["min_p"],
                    presence_penalty=base_params["presence_penalty"],
                    repetition_penalty=base_params["repetition_penalty"],
                    max_tokens=base_params["max_tokens"],
                    stop=base_params["stop"],
                    api_key=args.api_key,
                    timeout=args.request_timeout,
                    retries=args.server_request_retries,
                )
                for j, generation_info in enumerate(generation_infos):
                    text = generation_info["text"]
                    finish_reason = generation_info.get("finish_reason")
                    output_tokens = generation_info.get("completion_tokens")
                    reached_max_tokens = _reached_max_tokens(
                        finish_reason,
                        output_tokens,
                        args.max_tokens,
                    )
                    total_generations += 1
                    if reached_max_tokens:
                        max_token_generations += 1

                    lang = (args.language_tgt_type or args.language_src_type or "")
                    if not text:
                        logger.warning(
                            "Empty generation: task_id=%s src=%s tgt=%s output_file=%s",
                            task_ids[j],
                            args.language_src_type,
                            args.language_tgt_type,
                            args.output_file,
                        )
                    truncated = _truncate_to_finished(text, lang, args.dataset)
                    cleaned = cleanup_code(truncated, language_type=lang or "", dataset=args.dataset)

                    # サーバーモードではトークン列やスコアが得られない場合が多いので既定値を設定
                    record = {
                        "task_id": task_ids[j],
                        "prompt": prompts[j],
                        "generation": cleaned,
                        "scores": 0.0,
                        "finish": 2 if finish_reason in ("stop", "eos") else 1,
                        "output": [],
                        "finish_reason": finish_reason,
                        "output_tokens": output_tokens,
                        "reached_max_tokens": reached_max_tokens,
                    }
                    outfile.write(json.dumps(record, ensure_ascii=False) + "\n")

            n_done += len(batch)
            outfile.flush()
            elapsed = time.perf_counter() - start_time
            logger.info(f"Processed {n_done}/{total} tasks (elapsed: {elapsed:.1f}s)")

    os.replace(tmp_output_file, args.output_file)

    max_token_rate = max_token_generations / total_generations if total_generations else 0.0
    logger.info(
        "max_tokens reached: %d/%d (%.4f)",
        max_token_generations,
        total_generations,
        max_token_rate,
    )
    print(
        "max_tokens reached:",
        max_token_generations,
        "/",
        total_generations,
        f"({max_token_rate:.4f})",
    )

if __name__ == "__main__":
    main()

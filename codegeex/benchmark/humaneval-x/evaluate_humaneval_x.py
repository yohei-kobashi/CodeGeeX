import os
import sys
import fire
import json
import gzip
import regex
import numpy as np

from typing import *
from tqdm.auto import tqdm
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from codegeex.benchmark.utils import read_dataset, IMPORT_HELPER
from codegeex.benchmark.metric import estimate_pass_at_k
from codegeex.benchmark.execution import check_correctness

LANGUAGE_NAME = {
    "cpp"   : "CPP",
    "go"    : "Go",
    "java"  : "Java",
    "js"    : "JavaScript",
    "python": "Python",
    "rust"  : "Rust",
}


def _extract_translation_target(path: str) -> Optional[str]:
    """Extract target language token from filename/path.

    Supports patterns like:
      - "...-to-<tgt>-..."
      - "..._to_<tgt>..."
    Returns normalized language in {cpp, go, java, js, python, rust} or None.
    """
    s = path.lower()
    # Try hyphen/underscore variants: capture letters and '+' (for c++)
    m = regex.search(r"[-_]to[-_]([a-z\+]+)", s)
    if not m:
        return None
    raw = m.group(1)
    # Normalization map for common aliases
    alias = {
        "javascript": "js",
        "js": "js",
        "python": "python",
        "py": "python",
        "c++": "cpp",
        "cplusplus": "cpp",
        "cpp": "cpp",
        "go": "go",
        "java": "java",
        "rust": "rust",
    }
    # Direct match first
    if raw in alias:
        return alias[raw]
    # Fallback: substring match against known aliases
    for cand in ["cpp", "c++", "cplusplus", "go", "java", "javascript", "js", "python", "py", "rust"]:
        if cand in raw:
            return alias.get(cand, cand)
    return None


def process_humaneval_test(sample, problems, example_test=False):
    task_id = sample["task_id"]
    language = task_id.split("/")[0].lower()

    prompt = sample["prompt"]
    if example_test and "example_test" in problems[task_id] and problems[task_id]["example_test"] != "":
        test = problems[task_id]["example_test"]
    else:
        test = problems[task_id]["test"]
    code = sample["generation"]

    # Pre-process for different languages
    if language == "python":
        code_ = []
        for line in code.split("\n"):
            if (len(line.strip()) > 0 and line[0] != ' ' and line[0] != '\t'):
                break
            code_.append(line)
        code = "\n".join(code_)
        test_setup = "\n".join(IMPORT_HELPER["python"]) + "\n"
        test_string = test_setup + prompt + code + "\n" + test + "\n"
    elif language == "cpp":
        test_set_up = ""
        for s in IMPORT_HELPER["cpp"]:
            if s not in prompt:
                test_set_up += s + "\n"
        test_string = test_set_up + "\n" + prompt + code + "\n" + test
    elif language == "java":
        test_string = prompt + code + "\n" + test
    elif language == "js" or language == "javascript":
        test_string = prompt + code + "\n" + test
    elif language == "go":
        import_string = problems[task_id]["import"]
        prompt = prompt.replace(import_string, "")
        if example_test and "example_test" in problems[task_id]:
            test = problems[task_id]["example_test"]
        else:
            test = problems[task_id]["test"]
        test_setup = problems[task_id]["test_setup"]
        other_pkgs = []
        for pkg in IMPORT_HELPER["go"]:
            if pkg not in test_setup:
                p = pkg.split("/")[-1]
                if p + "." in code:
                    other_pkgs.append(f"\"{pkg}\"")
        if other_pkgs:
            import_other_pkgs = "import (\n" + "    ".join([p + "\n" for p in other_pkgs]) + ")"
            test_string = test_setup + "\n" + import_other_pkgs + "\n" + prompt + code + "\n" + test
        else:
            test_string = test_setup + "\n" + prompt + code + "\n" + test
    elif language == "rust":
        main = "\nfn main(){ \n } \n"
        declaration = problems[task_id]["declaration"]
        test_string = main + declaration + prompt + code + test

    return test_string


def stream_jsonl_all(filename: str) -> Iterable[Dict]:
    results = []
    if filename.endswith(".gz"):
        fp = gzip.open(open(filename, "rb"), "rt")
    else:
        fp = open(filename, "r")
    for line in fp:
        if any(not x.isspace() for x in line):
            results.append(json.loads(line))
    fp.close()

    return results


def _lang_from_problem_file(problem_file: str) -> Optional[str]:
    """Infer target language key from problem_file path.

    Expected patterns like .../humaneval_<lang>.jsonl.gz
    Returns normalized key in {cpp, go, java, js, python, rust} or None.
    """
    s = (problem_file or "").lower()
    m = regex.search(r"humaneval[_-]([a-z0-9\+]+)\.jsonl(\.gz)?$", s)
    if not m:
        return None
    raw = m.group(1)
    return {
        "javascript": "js",
        "js": "js",
        "python": "python",
        "py": "python",
        "c++": "cpp",
        "cplusplus": "cpp",
        "cpp": "cpp",
        "go": "go",
        "java": "java",
        "rust": "rust",
    }.get(raw, raw)


def evaluate_functional_correctness(
        input_file: str = None,
        tmp_dir: str = "./",
        n_workers: int = 32,
        timeout: float = 500.0,
        problem_file: str = "../data/humaneval_python.jsonl.gz",
        out_dir: str = None,
        k: List[int] = [1, 10, 100],
        test_groundtruth: bool = False,
        example_test: bool = False,
):
    if example_test:
        print("Example test...", file=sys.stderr)

    problems = read_dataset(problem_file,
                            dataset_type="humaneval")
    sample_jsonl = stream_jsonl_all(input_file)

    if example_test:
        suffix = "_example_test.jsonl"
    else:
        suffix = "_results.jsonl"
    if out_dir is not None:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_file = os.path.join(out_dir, input_file.split('/')[-1].replace(".jsonl", suffix))
    else:
        out_file = os.path.join(input_file.replace(".jsonl", suffix))

    # Ground truth mode is now controlled only by the explicit flag 'test_groundtruth'.
    # Do not auto-enable based on input path.

    tgt_lang = _extract_translation_target(input_file)
    translation_mode = tgt_lang is not None

    with ThreadPoolExecutor(max_workers=n_workers) as executor:

        futures = []
        completion_id = Counter()
        n_samples = 0
        results = defaultdict(list)

        if test_groundtruth:
            print("Testing ground truth...", file=sys.stderr)
            for sample in tqdm(problems.values()):
                task_id = sample["task_id"]
                lang = task_id.split("/")[0].lower()
                if lang == "javascript":
                    lang = "js"
                tmp_dir_ = os.path.join(tmp_dir, lang, "evaluation")
                sample["generation"] = sample["canonical_solution"]
                sample["test_code"] = process_humaneval_test(sample, problems, example_test)
                if sample["test_code"] is None:
                    continue
                args = (task_id, sample, lang, timeout, tmp_dir_, completion_id[task_id])
                future = executor.submit(check_correctness, *args)
                futures.append(future)
                completion_id[task_id] += 1
                n_samples += 1
        else:
            print("Reading samples...", file=sys.stderr)
            for sample in tqdm(sample_jsonl):
                task_id = sample["task_id"]
                lang = task_id.split("/")[0].lower()
                if translation_mode:
                    task_id = sample["task_id"].split("/")[-1]
                    lang = tgt_lang
                    # Safety: ensure lang is one of supported keys
                    if lang not in LANGUAGE_NAME:
                        for l in LANGUAGE_NAME:
                            if l in (lang or ""):
                                lang = l
                                break
                    task_id = f"{LANGUAGE_NAME[lang]}/{task_id}"
                if lang == "javascript":
                    lang = "js"
                tmp_dir_ = os.path.join(tmp_dir, lang, "evaluation")
                sample["task_id"] = task_id
                # Support both 'generation' and 'generated' fields in inputs
                if "generation" not in sample and "generated" in sample:
                    sample["generation"] = sample["generated"]
                sample["test_code"] = process_humaneval_test(sample, problems, example_test)
                if sample["test_code"] is None:
                    continue
                if "completion_id" in sample:
                    completion_id_ = sample["completion_id"]
                else:
                    completion_id_ = completion_id[task_id]
                args = (task_id, sample, lang, timeout, tmp_dir_, completion_id_)
                future = executor.submit(check_correctness, *args)
                futures.append(future)
                completion_id[task_id] += 1
                n_samples += 1

        print(completion_id, file=sys.stderr)
        if len(completion_id) == len(problems):
            evaluate_pass_at_k = True
        else:
            evaluate_pass_at_k = False

        print("Running test suites...", file=sys.stderr)
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            results[result["task_id"]].append((result["completion_id"], result))

    # Calculate pass@k.
    total, correct = [], []
    for result in results.values():
        passed = [r[1]["passed"] for r in result]
        total.append(len(passed))
        correct.append(sum(passed))
    total = np.array(total)
    correct = np.array(correct)
    pass_at_k = {}
    if evaluate_pass_at_k:
        ks = k
        pass_at_k = {f"pass@{kk}": estimate_pass_at_k(total, correct, kk).mean()
                     for kk in ks if (total >= kk).all()}
    else:
        # Keep totals on stderr for diagnostics without polluting stdout CSV
        print("Total:", np.sum(total), file=sys.stderr)
        print("Correct:", np.sum(correct), file=sys.stderr)

    print("Writing to: ", out_file, file=sys.stderr)
    if out_file.endswith(".gz"):
        fp = gzip.GzipFile(fileobj=open(out_file, "wb"), mode="wb")
        for res in results.values():
            for r in res:
                fp.write((json.dumps(r[1]) + "\n").encode("utf-8"))
    else:
        fp = open(out_file, 'w')
        for res in results.values():
            for r in res:
                fp.write(json.dumps(r[1]) + "\n")
    fp.close()

    # Compose CSV row for stdout; shell will add header.
    # language: infer from problem_file; input_file: as provided; metrics: formatted or blank.
    lang_key = _lang_from_problem_file(problem_file) or ""
    def fmt(v):
        try:
            return f"{float(v):.6f}"
        except Exception:
            return ""
    csv_vals = [
        lang_key,
        input_file or "",
        fmt(pass_at_k.get("pass@1", "")),
        fmt(pass_at_k.get("pass@10", "")),
        fmt(pass_at_k.get("pass@100", "")),
    ]
    # Only CSV row to stdout
    print(",".join(csv_vals))
    print("Evaluation finished.", file=sys.stderr)


def main():
    fire.Fire(evaluate_functional_correctness)


if __name__ == "__main__":
    sys.exit(main())

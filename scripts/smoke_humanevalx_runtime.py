#!/usr/bin/env python3

"""
Minimal runtime smoke tests for each language executor used by HumanEval-X.

This script runs a tiny "pass" and "fail" sample per language using the same
check_correctness() executor as the evaluator, without relying on dataset files.

How to run inside Singularity/Apptainer (recommended):
  singularity exec -B <repo_root>:/workspace \
    codegeex/benchmark/humaneval-x/humanevalx.sif \
    python3 /workspace/scripts/smoke_humanevalx_runtime.py

Expected output: a short CSV with columns:
  language,case,passed,result

Where:
  - case: pass_case / fail_case (expected outcome)
  - passed: true/false from executor
  - result: executor diagnostic string
"""

import os
import sys
from typing import Dict, Tuple

from codegeex.benchmark.execution import check_correctness


LANGS = ["python", "js", "go", "cpp", "rust", "java"]


def samples_for_language(lang: str) -> Tuple[Dict, Dict]:
    """Return (pass_sample, fail_sample) with keys: task_id, prompt, generation, test_code."""
    if lang == "python":
        pass_code = """
def add(a, b):
    return a + b

assert add(1, 2) == 3
""".strip()
        fail_code = """
def add(a, b):
    return a + b

assert add(1, 2) == 4
""".strip()
    elif lang == "js":
        # Note: JS executor treats any stdout/stderr as failure. Emit nothing on success.
        pass_code = """
function add(a, b) { return a + b; }
if (add(1, 2) !== 3) { throw new Error("wrong"); }
""".strip()
        # Intentionally print to stdout to be considered failure by executor.
        fail_code = """
function add(a, b) { return a + b; }
if (add(1, 2) !== 4) { console.log("mismatch"); }
""".strip()
    elif lang == "go":
        pass_code = """
package main

import "testing"

func add(a, b int) int { return a + b }

func TestAdd(t *testing.T) {
    if add(1, 2) != 3 {
        t.Fatalf("wrong")
    }
}
""".strip()
        fail_code = """
package main

import "testing"

func add(a, b int) int { return a + b }

func TestAdd(t *testing.T) {
    if add(1, 2) != 4 {
        t.Fatalf("wrong")
    }
}
""".strip()
    elif lang == "cpp":
        pass_code = """
int add(int a, int b) { return a + b; }
int main() {
    if (add(1, 2) != 3) return 1;
    return 0;
}
""".strip()
        fail_code = """
int add(int a, int b) { return a + b; }
int main() {
    if (add(1, 2) != 4) return 1;
    return 0;
}
""".strip()
    elif lang == "rust":
        pass_code = """
pub fn add(a: i32, b: i32) -> i32 { a + b }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_add() {
        assert_eq!(add(1, 2), 3);
    }
}
""".strip()
        fail_code = """
pub fn add(a: i32, b: i32) -> i32 { a + b }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_add() {
        assert_eq!(add(1, 2), 4);
    }
}
""".strip()
    elif lang == "java":
        # Note: current executor may not run Java (depending on version); this still compiles.
        pass_code = """
public class Main {
    static int add(int a, int b) { return a + b; }
    public static void main(String[] args) {
        if (add(1, 2) != 3) throw new AssertionError("wrong");
    }
}
""".strip()
        fail_code = """
public class Main {
    static int add(int a, int b) { return a + b; }
    public static void main(String[] args) {
        if (add(1, 2) != 4) throw new AssertionError("wrong");
    }
}
""".strip()
    else:
        raise ValueError(f"Unknown language: {lang}")

    def mk(task: str, code: str) -> Dict:
        return {
            "task_id": task,
            "prompt": "",
            "generation": "",
            "test_code": code,
        }

    task_prefix = {
        "python": "Python",
        "js": "JavaScript",
        "go": "Go",
        "cpp": "CPP",
        "rust": "Rust",
        "java": "Java",
    }[lang]
    return (
        mk(f"{task_prefix}/smoketest_pass", pass_code),
        mk(f"{task_prefix}/smoketest_fail", fail_code),
    )


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    tmp_root = os.path.join(repo_root, "codegeex", "benchmark", "humaneval-x")
    timeout = 5.0

    print("language,case,passed,result")
    for lang in LANGS:
        ok, ng = samples_for_language(lang)
        # pass case
        r_ok = check_correctness(ok["task_id"], ok, lang, timeout=timeout, tmp_dir=tmp_root, completion_id=0)
        print(f"{lang},pass_case,{str(r_ok['passed']).lower()},{r_ok['result']}")
        # fail case
        r_ng = check_correctness(ng["task_id"], ng, lang, timeout=timeout, tmp_dir=tmp_root, completion_id=1)
        print(f"{lang},fail_case,{str(r_ng['passed']).lower()},{r_ng['result']}")


if __name__ == "__main__":
    sys.exit(main())


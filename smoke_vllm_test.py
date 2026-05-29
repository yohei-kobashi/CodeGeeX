"""vLLM smoke test for merged HF models. Run inside the verl_arm.sif singularity."""
import os


def main():
    from vllm import LLM, SamplingParams

    model_path = os.environ["MODEL_PATH"]
    max_model_len = int(os.environ.get("MAX_MODEL_LEN", "4096"))
    gpu_util = float(os.environ.get("GPU_UTIL", "0.6"))

    print(f"[smoke] loading: {model_path}")
    print(f"[smoke] max_model_len={max_model_len} gpu_util={gpu_util}")

    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_util,
        trust_remote_code=True,
        enforce_eager=True,
    )

    tok = llm.get_tokenizer()
    messages = [
        {
            "role": "user",
            "content": (
                "言語モデルについて教えて"
            ),
        }
    ]
    try:
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception as e:
        print(f"[smoke] chat_template failed ({e}); falling back to plain prompt")
        prompt = messages[0]["content"]

    print("[smoke] prompt:")
    print(prompt[:8192])
    print("..." if len(prompt) > 8192 else "")

    params = SamplingParams(temperature=0.2, top_p=0.95, max_tokens=8192)
    outputs = llm.generate([prompt], params)

    print("\n[smoke] generated:")
    print(outputs[0].outputs[0].text)
    print("\n[smoke] ok")


if __name__ == "__main__":
    main()

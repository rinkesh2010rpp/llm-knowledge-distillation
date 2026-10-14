"""Let a model make each logged decision itself, with vLLM, for the judge to score.

For every example in a split, the model gets exactly what DeepSeek got (system
prompt, history, tools) and generates its own next step. Prompts are rendered
with the same chat template as training.

    modal run --detach 06_generate.py --model base --split test
    modal run --detach 06_generate.py --model run1 --split test

`--model base` serves the downloaded model; any other name serves
/vol/models/<name>-merged (from 05_merge_adapter.py). Output:
/vol/eval/<split>/<model>.jsonl.
"""
import modal

MODEL_DIR = "/vol/models/Qwen3.6-35B-A3B"
MAX_SEQ = 98304

app = modal.App("sudarshana-distill-generate")
volume = modal.Volume.from_name("sudarshana-distill", create_if_missing=True)
# CUDA devel base: vLLM JIT-compiles some kernels (via nvcc) during warm-up.
image = modal.Image.from_registry("nvidia/cuda:13.0.3-devel-ubuntu24.04", add_python="3.12").pip_install("vllm")


@app.function(image=image, gpu="H200", volumes={"/vol": volume}, memory=65536, timeout=90 * 60)
def generate(model: str, split: str, max_tokens: int, limit: int):
    import json
    import os
    import time

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    start = time.time()
    path = MODEL_DIR if model == "base" else f"/vol/models/{model}-merged"
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    with open(f"/vol/data/{split}.jsonl", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    if limit:
        rows = rows[:limit]
    # The file is grouped by turn, so consecutive prompts share long prefixes
    # and vLLM's prefix cache skips most of the repeated prefill.
    prompts = []
    for ex in rows:
        text = tok.apply_chat_template(ex["messages"], tools=ex["tools"], add_generation_prompt=True, tokenize=False)
        prompts.append(TokensPrompt(prompt_token_ids=tok(text, add_special_tokens=False)["input_ids"]))

    llm = LLM(
        model=path,
        tokenizer=MODEL_DIR,
        max_model_len=MAX_SEQ + max_tokens,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.92,
        limit_mm_per_prompt={"image": 0, "video": 0},  # text only
        seed=0,
    )
    print(f"[gen] {model}: loaded in {(time.time() - start) / 60:.1f} min, {len(prompts)} prompts")
    # Qwen's recommended sampling for thinking mode.
    params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=max_tokens, seed=0, skip_special_tokens=False)
    t0 = time.time()
    outputs = llm.generate(prompts, params)
    gen_s = time.time() - t0

    out_dir = f"/vol/eval/{split}"
    os.makedirs(out_dir, exist_ok=True)
    suffix = f"-first{limit}" if limit else ""
    new_tokens = 0
    with open(f"{out_dir}/{model}{suffix}.jsonl", "w", encoding="utf-8") as fh:
        for ex, out in zip(rows, outputs):
            o = out.outputs[0]
            new_tokens += len(o.token_ids)
            fh.write(json.dumps({
                "id": ex["id"], "model": model, "text": o.text, "n_tokens": len(o.token_ids),
                "finish_reason": o.finish_reason,
            }, ensure_ascii=False) + "\n")
    volume.commit()
    print(f"[gen] {model}: {len(rows)} prompts in {gen_s / 60:.1f} min, {new_tokens:,} new tokens "
          f"({new_tokens / gen_s:.0f} tok/s), total {(time.time() - start) / 60:.1f} min")


@app.local_entrypoint()
def main(model: str = "base", split: str = "test", max_tokens: int = 8192, limit: int = 0):
    call = generate.spawn(model, split, max_tokens, limit)
    print(f"Spawned {call.object_id}")

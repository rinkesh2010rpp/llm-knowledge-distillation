"""Compare the base model with a trained LoRA on held-out val examples.

Two checks, both run for the base model (LoRA off) and the trained model:

1. Loss on DeepSeek's reply, split into reasoning (<think>), reply text and
   tool calls, so a flat average can't hide where training helped.
2. Next action: each model generates its own reply for a few inputs, compared
   with what DeepSeek did (reply vs. tool call, same tool, same arguments).

    modal run --detach 04_validate.py --run-name run1

Results go to /vol/runs/<run-name>/validate.json and the logs.
"""
import modal

MODEL_DIR = "/vol/models/Qwen3.6-35B-A3B"
MAX_SEQ = 98304
TIMEOUT = 45 * 60  # hard cap on the whole job

app = modal.App("sudarshana-distill-validate")
volume = modal.Volume.from_name("sudarshana-distill", create_if_missing=True)
# Same image as 03_train_sft.py.
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.3-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "build-essential", "ninja-build")
    .pip_install("unsloth", "unsloth_zoo", "flash-linear-attention", "torchvision", "pillow", "packaging", "wheel")
    .env({"TORCH_CUDA_ARCH_LIST": "9.0", "MAX_JOBS": "8", "CC": "gcc", "CXX": "g++"})
    .run_commands("pip install causal-conv1d --no-build-isolation")
)

SEGMENTS = ("reasoning", "text", "tool_calls")


@app.function(image=image, gpu="H200", volumes={"/vol": volume}, memory=65536, timeout=TIMEOUT)
def validate(run_name: str, n_loss: int, n_gen: int, max_new_tokens: int, gen_minutes: int):
    import json
    import random
    import re
    import time

    import torch
    from unsloth import FastModel
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    run_dir = f"/vol/runs/{run_name}"
    with open(f"{run_dir}/adapter/adapter_config.json") as fh:
        adapter_cfg = json.load(fh)

    model, _ = FastModel.from_pretrained(
        model_name=MODEL_DIR, max_seq_length=MAX_SEQ, load_in_4bit=False, load_in_16bit=True, full_finetuning=False,
    )
    model = FastModel.get_peft_model(
        model,
        r=adapter_cfg["r"],
        lora_alpha=adapter_cfg["lora_alpha"],
        lora_dropout=0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=MAX_SEQ,
    )
    set_peft_model_state_dict(model, load_file(f"{run_dir}/adapter/adapter_model.safetensors"))
    model.eval()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)

    # The same 30 val examples 03_train_sft.py scores at the end of each run.
    with open("/vol/data/val.jsonl", encoding="utf-8") as fh:
        val = [json.loads(line) for line in fh]
    val.sort(key=lambda ex: ex["id"])
    random.Random(1).shuffle(val)

    def encode(ex):
        prompt = tok.apply_chat_template(ex["messages"], tools=ex["tools"], add_generation_prompt=True, tokenize=False)
        full = tok.apply_chat_template(ex["messages"] + [ex["target"]], tools=ex["tools"], tokenize=False)
        if not full.startswith(prompt):
            return None
        reply = full[len(prompt):]
        prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        enc = tok(reply, add_special_tokens=False, return_offsets_mapping=True)
        if len(prompt_ids) + len(enc["input_ids"]) > MAX_SEQ:
            return None
        # Label each reply token by the part of the answer it starts in.
        think_end = reply.find("</think>")
        text_start = think_end + len("</think>") if think_end >= 0 else 0
        tool_start = reply.find("<tool_call>", text_start)
        tool_start = len(reply) if tool_start < 0 else tool_start
        end = reply.rfind("<|im_end|>")
        end = len(reply) if end < 0 else end
        segs = []
        for s, e in enc["offset_mapping"]:
            if think_end >= 0 and s < think_end:
                segs.append("reasoning")
            elif text_start <= s < tool_start and reply[s:e].strip():
                segs.append("text")
            elif tool_start <= s < end:
                segs.append("tool_calls")
            else:
                segs.append(None)  # </think>, whitespace, <|im_end|>
        return prompt_ids, enc["input_ids"], segs

    def seg_loss(prompt_ids, reply_ids, segs, which):
        n = sum(1 for s in segs if s == which)
        if not n:
            return None, 0
        ids = torch.tensor([prompt_ids + reply_ids], device="cuda")
        labels = [-100] * len(prompt_ids) + [t if s == which else -100 for t, s in zip(reply_ids, segs)]
        with torch.no_grad():
            loss = model(input_ids=ids, labels=torch.tensor([labels], device="cuda")).loss.item()
        return loss, n

    start = time.time()
    encoded = [(ex, encode(ex)) for ex in val[:n_loss]]
    encoded = [(ex, e) for ex, e in encoded if e is not None]

    # --- 1. loss by part of the answer ---
    totals = {m: {s: [0.0, 0] for s in SEGMENTS} for m in ("base", "trained")}
    for i, (ex, (prompt_ids, reply_ids, segs)) in enumerate(encoded):
        for which in SEGMENTS:
            with model.disable_adapter():
                loss, n = seg_loss(prompt_ids, reply_ids, segs, which)
            if loss is not None:
                totals["base"][which][0] += loss * n
                totals["base"][which][1] += n
                loss, n = seg_loss(prompt_ids, reply_ids, segs, which)
                totals["trained"][which][0] += loss * n
                totals["trained"][which][1] += n
        print(f"[loss] {i + 1}/{len(encoded)} done, {(time.time() - start) / 60:.1f} min")

    loss_summary = {}
    for m, by_seg in totals.items():
        row = {s: round(v[0] / v[1], 4) if v[1] else None for s, v in by_seg.items()}
        all_n = sum(v[1] for v in by_seg.values())
        row["overall"] = round(sum(v[0] for v in by_seg.values()) / all_n, 4)
        row["tokens"] = {s: v[1] for s, v in by_seg.items()}
        loss_summary[m] = row
    print(json.dumps({"loss": loss_summary}, indent=1))

    # --- 2. next action: generate and compare with DeepSeek ---
    def parse(text):
        finished = "</think>" in text
        body = text.split("</think>", 1)[1] if finished else ""
        calls = []
        for m in re.finditer(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>", body, re.S):
            args = dict(re.findall(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", m.group(2), re.S))
            calls.append({"name": m.group(1).strip(), "args": {k: v.strip() for k, v in args.items()}})
        reply_text = body.split("<tool_call>")[0].replace("<|im_end|>", "").strip()
        return finished, calls, reply_text

    def as_str(v):
        return v.strip() if isinstance(v, str) else json.dumps(v, ensure_ascii=False)

    def compare(ex, finished, calls, reply_text):
        want = ex["target"].get("tool_calls") or []
        names = {t["function"]["name"] for t in ex["tools"] or []}
        want_tool = bool(want)
        got_tool = bool(calls)
        row = {
            "finished_thinking": finished,
            "valid": finished and (got_tool or bool(reply_text)) and all(c["name"] in names for c in calls),
            "same_action_type": finished and want_tool == got_tool,
        }
        # Tool metrics only mean something when DeepSeek called a tool (None = not applicable).
        row["same_first_tool"] = bool(calls and want[0]["function"]["name"] == calls[0]["name"]) if want_tool else None
        if row["same_first_tool"]:
            want_args = {k: as_str(v) for k, v in want[0]["function"]["arguments"].items()}
            row["same_first_args"] = want_args == calls[0]["args"]
        else:
            row["same_first_args"] = False if want_tool else None
        return row

    def summarize_call(calls, reply_text):
        if calls:
            c = calls[0]
            return f"{c['name']}({json.dumps(c['args'], ensure_ascii=False)[:200]})" + (f" +{len(calls) - 1}" if len(calls) > 1 else "")
        return "REPLY: " + reply_text[:200]

    FastModel.for_inference(model)
    gen_start = time.time()
    gen_rows = []
    for ex, (prompt_ids, _, _) in encoded[:n_gen]:
        if time.time() - gen_start > gen_minutes * 60:
            print("[gen] time budget reached")
            break
        want = ex["target"].get("tool_calls") or []
        target_calls = [{"name": t["function"]["name"], "args": {k: as_str(v) for k, v in t["function"]["arguments"].items()}} for t in want]
        row = {"id": ex["id"], "kind": ex["kind"], "deepseek": summarize_call(target_calls, ex["target"].get("content") or "")}
        for m in ("base", "trained"):
            t0 = time.time()
            ids = torch.tensor([prompt_ids], device="cuda")
            torch.manual_seed(0)
            if m == "base":
                with model.disable_adapter():
                    out = model.generate(input_ids=ids, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.6, top_p=0.95, top_k=20)
            else:
                out = model.generate(input_ids=ids, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.6, top_p=0.95, top_k=20)
            text = tok.decode(out[0][len(prompt_ids):], skip_special_tokens=False)
            finished, calls, reply_text = parse(text)
            row[m] = {**compare(ex, finished, calls, reply_text), "action": summarize_call(calls, reply_text), "new_tokens": len(out[0]) - len(prompt_ids), "seconds": round(time.time() - t0, 1)}
        gen_rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    gen_summary = {}
    for m in ("base", "trained"):
        rows = [r[m] for r in gen_rows]
        gen_summary[m] = {}
        for k in ("finished_thinking", "valid", "same_action_type", "same_first_tool", "same_first_args"):
            applicable = [r[k] for r in rows if r[k] is not None]
            gen_summary[m][k] = f"{sum(applicable)}/{len(applicable)}"
    result = {"run": run_name, "loss": loss_summary, "next_action": gen_summary, "examples": gen_rows, "minutes": round((time.time() - start) / 60, 1)}
    print(json.dumps({"loss": loss_summary, "next_action": gen_summary}, indent=1))
    with open(f"{run_dir}/validate.json", "w") as fh:
        json.dump(result, fh, indent=1, ensure_ascii=False)
    volume.commit()


@app.local_entrypoint()
def main(run_name: str = "run1", n_loss: int = 30, n_gen: int = 10, max_new_tokens: int = 2048, gen_minutes: int = 20):
    call = validate.spawn(run_name, n_loss, n_gen, max_new_tokens, gen_minutes)
    print(f"Spawned {call.object_id}")

"""LoRA SFT of Qwen3.6-35B-A3B on Sudarshana's logged calls, time-boxed and resumable.

Each run trains for at most GPU_SECONDS, then saves the LoRA adapter, optimizer
and data position to the Volume and measures loss on a few val examples. The
next run with the same --run-name continues where the last one stopped.

    modal run 03_train_sft.py --run-name run1

Needs the base model (02_download_model.py) and the dataset on the
`sudarshana-distill` Volume:

    modal volume put sudarshana-distill data/train.jsonl /data/train.jsonl
    modal volume put sudarshana-distill data/val.jsonl /data/val.jsonl
"""
import modal

MODEL_DIR = "/vol/models/Qwen3.6-35B-A3B"
GPU_SECONDS = 2 * 3600  # training stops here...
POST_SECONDS = 15 * 60  # ...and saving + val check must finish within this
CHECKPOINT_EVERY = 15 * 60  # Modal can preempt; lose at most this much
MAX_SEQ = 98304

app = modal.App("sudarshana-distill-train")
volume = modal.Volume.from_name("sudarshana-distill", create_if_missing=True)
image = (
    # CUDA devel base: causal-conv1d compiles CUDA kernels. Without it, Qwen3.6's
    # linear-attention conv falls back to a much slower PyTorch path.
    modal.Image.from_registry("nvidia/cuda:13.0.3-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "build-essential", "ninja-build")
    .pip_install("unsloth", "unsloth_zoo", "flash-linear-attention", "torchvision", "pillow", "packaging", "wheel")
    .env({"TORCH_CUDA_ARCH_LIST": "9.0", "MAX_JOBS": "8", "CC": "gcc", "CXX": "g++"})  # H200 = sm_90
    .run_commands("pip install causal-conv1d --no-build-isolation")
)


@app.function(
    image=image,
    gpu="H200",
    volumes={"/vol": volume},
    memory=65536,
    timeout=GPU_SECONDS + POST_SECONDS,  # hard cap: Modal kills the run here
)
def train(run_name: str, lr: float, grad_accum: int, rank: int, val_examples: int):
    import json
    import os
    import random
    import time

    start = time.time()
    run_dir = f"/vol/runs/{run_name}"
    os.makedirs(run_dir, exist_ok=True)
    state_path = f"{run_dir}/state.json"
    state = {"position": 0, "step": 0, "tokens": 0, "call_id": None, "call_seconds": 0.0}
    if os.path.exists(state_path):
        with open(state_path) as fh:
            state = json.load(fh)
    # A new `modal run` gets a fresh budget. If Modal preempts the container and
    # restarts this same call, only the time it hadn't used yet is left.
    call_id = modal.current_function_call_id()
    if state["call_id"] != call_id:
        state["call_id"], state["call_seconds"] = call_id, 0.0
    budget = max(0.0, GPU_SECONDS - state["call_seconds"])
    print(f"[run] {run_name}: resuming at example {state['position']}, step {state['step']}, budget {budget / 60:.0f} min")

    import torch
    from unsloth import FastModel
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    with open("/vol/data/train.jsonl", encoding="utf-8") as fh:
        train_lines = fh.readlines()
    with open("/vol/data/val.jsonl", encoding="utf-8") as fh:
        val_lines = fh.readlines()

    # Fixed order for the whole run, saved so later runs continue from the next
    # unseen example. Longest first: if 96K doesn't fit, we find out in a minute.
    order_path = f"{run_dir}/order.json"
    if os.path.exists(order_path):
        with open(order_path) as fh:
            order = json.load(fh)
    else:
        lengths = [json.loads(line)["n_tokens"] for line in train_lines]
        order = list(range(len(train_lines)))
        random.Random(0).shuffle(order)
        longest = max(order, key=lambda i: lengths[i])
        order.remove(longest)
        order.insert(0, longest)
        with open(order_path, "w") as fh:
            json.dump(order, fh)

    model, _ = FastModel.from_pretrained(
        model_name=MODEL_DIR,
        max_seq_length=MAX_SEQ,
        load_in_4bit=False,  # Unsloth: no QLoRA for Qwen3.5/3.6
        load_in_16bit=True,
        full_finetuning=False,
    )
    model = FastModel.get_peft_model(
        model,
        r=rank,
        lora_alpha=rank,
        lora_dropout=0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=MAX_SEQ,
    )
    # The chat template we validated the dataset against (tools + reasoning).
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    # Constant LR after a short warmup: there's no fixed end to anneal toward
    # when training runs in resumable 2-hour slices.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / 10))

    if os.path.exists(f"{run_dir}/adapter/adapter_model.safetensors"):
        set_peft_model_state_dict(model, load_file(f"{run_dir}/adapter/adapter_model.safetensors"))
        optimizer.load_state_dict(torch.load(f"{run_dir}/optimizer.pt"))
        scheduler.load_state_dict(torch.load(f"{run_dir}/scheduler.pt"))
        print(f"[run] loaded checkpoint from step {state['step']}")

    def encode(line):
        ex = json.loads(line)
        prompt = tok.apply_chat_template(ex["messages"], tools=ex["tools"], add_generation_prompt=True, tokenize=False)
        full = tok.apply_chat_template(ex["messages"] + [ex["target"]], tools=ex["tools"], tokenize=False)
        if not full.startswith(prompt):
            return None
        # Tokenize prompt and reply separately: tokenizing `full` in one go can
        # merge tokens across the boundary (e.g. "<think>\n" + "\n</think>"), and
        # at serving time the model continues from exactly these prompt tokens.
        prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        reply_ids = tok(full[len(prompt):], add_special_tokens=False)["input_ids"]
        ids = prompt_ids + reply_ids
        if len(ids) > MAX_SEQ:
            return None
        # Loss only on DeepSeek's reply; the context is input, not something to learn.
        labels = [-100] * len(prompt_ids) + reply_ids
        return torch.tensor([ids], device="cuda"), torch.tensor([labels], device="cuda")

    def save(elapsed):
        model.save_pretrained(f"{run_dir}/adapter")
        torch.save(optimizer.state_dict(), f"{run_dir}/optimizer.pt")
        torch.save(scheduler.state_dict(), f"{run_dir}/scheduler.pt")
        with open(state_path, "w") as fh:
            json.dump({**state, "call_seconds": base_seconds + elapsed}, fh)
        volume.commit()

    base_seconds = state["call_seconds"]
    log = open(f"{run_dir}/log.jsonl", "a")
    model.train()
    last_ckpt = time.time()
    step_times = []
    while state["position"] < len(order):
        elapsed = time.time() - start
        avg_step = sum(step_times[-5:]) / len(step_times[-5:]) if step_times else 0
        if elapsed + avg_step > budget:
            print(f"[run] GPU budget reached at {elapsed / 60:.1f} min")
            break
        t0 = time.time()
        losses, tokens, skipped = [], 0, 0
        optimizer.zero_grad(set_to_none=True)
        taken = 0
        while taken < grad_accum and state["position"] < len(order):
            batch = encode(train_lines[order[state["position"]]])
            state["position"] += 1
            if batch is None:
                skipped += 1
                continue
            ids, labels = batch
            loss = model(input_ids=ids, labels=labels).loss
            (loss / grad_accum).backward()
            losses.append(loss.item())
            tokens += ids.shape[1]
            taken += 1
        if not losses:
            continue
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()
        state["step"] += 1
        state["tokens"] += tokens
        dt = time.time() - t0
        step_times.append(dt)
        entry = {
            "step": state["step"], "position": state["position"], "loss": sum(losses) / len(losses),
            "tokens": tokens, "tok_per_s": round(tokens / dt), "step_s": round(dt, 1),
            "max_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1), "skipped": skipped,
            "elapsed_min": round((time.time() - start) / 60, 1),
        }
        print(json.dumps(entry))
        log.write(json.dumps(entry) + "\n")
        log.flush()
        if time.time() - last_ckpt > CHECKPOINT_EVERY:
            save(time.time() - start)
            last_ckpt = time.time()
            print(f"[run] checkpoint at step {state['step']}")

    # --- post-processing: within POST_SECONDS of the training stop ---
    save(time.time() - start)
    print(f"[run] saved step {state['step']}, {state['position']}/{len(order)} examples, {state['tokens']:,} tokens")

    model.eval()
    val = [json.loads(line) for line in val_lines]
    val.sort(key=lambda ex: ex["id"])
    random.Random(1).shuffle(val)
    val_losses = []
    post_start = time.time()
    with torch.no_grad():
        for ex in val[:val_examples]:
            if time.time() - post_start > POST_SECONDS - 120:
                break
            batch = encode(json.dumps(ex, ensure_ascii=False))
            if batch is None:
                continue
            val_losses.append(model(input_ids=batch[0], labels=batch[1]).loss.item())
    if val_losses:
        result = {"step": state["step"], "val_loss": sum(val_losses) / len(val_losses), "val_n": len(val_losses)}
        print(json.dumps(result))
        with open(f"{run_dir}/val.jsonl", "a") as fh:
            fh.write(json.dumps(result) + "\n")
        volume.commit()


@app.local_entrypoint()
def main(run_name: str = "run1", lr: float = 1e-4, grad_accum: int = 8, rank: int = 16, val_examples: int = 30):
    # spawn, not remote: with `modal run --detach` the run keeps going after this
    # local process exits, so a closed terminal can't cancel it.
    call = train.spawn(run_name, lr, grad_accum, rank, val_examples)
    print(f"Spawned {call.object_id}. Follow with: modal app logs <app-id>")

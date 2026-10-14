"""Merge a run's LoRA into the base model, so vLLM can serve it as a plain model.

vLLM's LoRA support for MoE expert weights is the least mature part, so we
serve a merged copy instead: /vol/models/<run-name>-merged.

    modal run --detach 05_merge_adapter.py --run-name run1
"""
import modal

MODEL_DIR = "/vol/models/Qwen3.6-35B-A3B"
MAX_SEQ = 98304

app = modal.App("sudarshana-distill-merge")
volume = modal.Volume.from_name("sudarshana-distill", create_if_missing=True)
# Same image as 03_train_sft.py.
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.3-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "build-essential", "ninja-build")
    .pip_install("unsloth", "unsloth_zoo", "flash-linear-attention", "torchvision", "pillow", "packaging", "wheel")
    .env({"TORCH_CUDA_ARCH_LIST": "9.0", "MAX_JOBS": "8", "CC": "gcc", "CXX": "g++"})
    .run_commands("pip install causal-conv1d --no-build-isolation")
)


@app.function(image=image, gpu="H200", volumes={"/vol": volume}, memory=131072, timeout=45 * 60)
def merge(run_name: str):
    import json
    import os
    import shutil

    from unsloth import FastModel
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    run_dir = f"/vol/runs/{run_name}"
    out_dir = f"/vol/models/{run_name}-merged"
    with open(f"{run_dir}/adapter/adapter_config.json") as fh:
        cfg = json.load(fh)

    model, processor = FastModel.from_pretrained(
        model_name=MODEL_DIR, max_seq_length=MAX_SEQ, load_in_4bit=False, load_in_16bit=True, full_finetuning=False,
    )
    model = FastModel.get_peft_model(
        model,
        r=cfg["r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        random_state=3407,
        max_seq_length=MAX_SEQ,
    )
    set_peft_model_state_dict(model, load_file(f"{run_dir}/adapter/adapter_model.safetensors"))
    model.save_pretrained_merged(out_dir, processor, save_method="merged_16bit")

    # Keep the exact chat template and tokenizer files we trained and validated with.
    for name in os.listdir(MODEL_DIR):
        if name.endswith((".json", ".jinja", ".txt")) and not name.startswith("model") and name != "config.json":
            shutil.copy(f"{MODEL_DIR}/{name}", f"{out_dir}/{name}")
    volume.commit()
    print(f"Merged {run_name} into {out_dir}: {sorted(os.listdir(out_dir))}")


@app.local_entrypoint()
def main(run_name: str = "run1"):
    call = merge.spawn(run_name)
    print(f"Spawned {call.object_id}")

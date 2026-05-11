import modal

# 1. Environment optimized for Unsloth & Speed
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "unsloth", 
        "torch", 
        "transformers", 
        "datasets", 
        "peft", 
        "trl"
    )
)

app = modal.App("mental-health-specialist-training")
volume = modal.Volume.from_name("mental-health-data")

@app.function(
    image=image,
    gpu="L4", # Best price/performance for 1.5B models
    volumes={"/data": volume},
    timeout=3600, 
)
def train():
    from unsloth import FastLanguageModel
    import torch
    from datasets import load_dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments

    # --- CONFIG ---
    max_seq_length = 2048
    model_name = "Qwen/Qwen2.5-1.5B-Instruct" 

    print("🚀 Loading Student Model in 4-bit...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        load_in_4bit = True, 
    )

    # Add LoRA adapters (The "Brain" of our specialist)
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, 
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
    )

    # --- DATA PREPARATION ---
    print("📖 Loading your fixed clinical data...")
    dataset = load_dataset("json", data_files="/data/distilled_training_data_v2.jsonl", split="train")

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        outputs      = examples["teacher_distillation"]
        texts = []
        for instruction, output in zip(instructions, outputs):
            # Formats it exactly as the teacher responded
            text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
            texts.append(text)
        return { "text" : texts, }

    dataset = dataset.map(formatting_prompts_func, batched = True)

    # --- TRAINING SETTINGS ---
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        args = TrainingArguments(
            per_device_train_batch_size = 4, 
            gradient_accumulation_steps = 4,
            warmup_steps = 10,
            max_steps = 120, # 120 steps is perfect for 2,500 rows
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            output_dir = "outputs",
            optim = "adamw_8bit",
        ),
    )

    print("🧠 Training starting... Watch the LOSS column!")
    trainer.train()

    # --- SAVE THE MODEL ---
    print("💾 Saving the final specialist model to Volume...")
    # This merges the LoRA adapters for a clean, ready-to-use model
    model.save_pretrained_merged("/data/mental_health_specialist_v1", tokenizer, save_method = "merged_16bit")
    volume.commit()
    print("✅ Done! Your specialist is at /data/mental_health_specialist_v1")

@app.local_entrypoint()
def main():
    train.remote()
import modal
import json
import os

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("vllm==0.6.3", "pandas", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("mental-health-distill-teacher")
volume = modal.Volume.from_name("mental-health-data")

@app.function(
    image=image,
    gpu="A100-80GB", 
    volumes={"/data": volume},
    timeout=7200,
    secrets=[modal.Secret.from_name("hf-secret")]
)
def run_distillation():
    from vllm import LLM, SamplingParams
    
    # 1. Load Teacher
    model_name = "Qwen/Qwen2.5-72B-Instruct-AWQ" 
    llm = LLM(
        model=model_name, 
        trust_remote_code=True,
        quantization="awq",
        gpu_memory_utilization=0.90,
        max_model_len=2048,
    )
    sampling_params = SamplingParams(temperature=0.4, max_tokens=800)

    # 2. THE FIX: Logic that actually separates the crisis from the noise
    all_raw_data = []
    seen_questions = set()
    
    dataset_files = [
        "/data/ShenLab_MentalChat16K.jsonl",
        "/data/Amod_mental_health_counseling_conversations.jsonl",
        "/data/UVSKKR_Ethical_Reasoning-in-Mental-Health-v1.jsonl"
    ]

    for file_path in dataset_files:
        if not os.path.exists(file_path):
            continue
            
        print(f"📖 Processing {file_path}...")
        with open(file_path, "r") as f:
            for line in f:
                item = json.loads(line)
                
                # --- EXPLICIT LOGIC ---
                # Prioritize 'input' (the patient) over 'instruction' (the persona)
                patient_msg = item.get("input") or item.get("context") or item.get("scenario")
                persona_msg = item.get("instruction")
                
                # If 'input' is empty, try using 'instruction' only if it's NOT the generic prompt
                if patient_msg and len(str(patient_msg)) > 10:
                    final_q = str(patient_msg)
                elif persona_msg and "You are a helpful" not in str(persona_msg):
                    final_q = str(persona_msg)
                else:
                    continue # Skip this row, it's just noise

                if final_q not in seen_questions:
                    all_raw_data.append({"instruction": final_q})
                    seen_questions.add(final_q)

    target_data = all_raw_data[:2500] 
    print(f"✅ Ready! Found {len(target_data)} high-quality patient scenarios.")

    # 3. PROMPT & RUN
    formatted_prompts = [
        f"<|im_start|>system\nYou are a clinical psychologist. Write a 'Clinical Rationale' then a 'Final Response'.<|im_end|>\n<|im_start|>user\n{item['instruction']}<|im_end|>\n<|im_start|>assistant\nRationale:"
        for item in target_data
    ]

    outputs = llm.generate(formatted_prompts, sampling_params)

    # 4. SAVE (This will overwrite the 'bad' file)
    distilled_results = [
        {"instruction": target_data[i]["instruction"], "teacher_distillation": "Rationale: " + out.outputs[0].text}
        for i, out in enumerate(outputs)
    ]

    with open("/data/distilled_training_data_v2.jsonl", "w") as f:
        for entry in distilled_results:
            f.write(json.dumps(entry) + "\n")
    
    volume.commit()
    print(f"✅ Success! Data is now strictly clinical.")

@app.local_entrypoint()
def main():
    run_distillation.remote()
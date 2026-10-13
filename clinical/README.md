# Mental Health AI Specialist — Fine-Tuning via Knowledge Distillation

A learning project that fine-tunes a small language model (Qwen 1.5B) into a clinical mental health specialist using knowledge distillation from a large teacher model (Qwen 72B). All training runs serverlessly on [Modal](https://modal.com).

Run every command below from this `clinical/` folder.

## How It Works

```
Raw datasets → Teacher distillation → Fine-tune student → Compare models
```

1. **Ingest** — Downloads three mental health datasets from Hugging Face
2. **Distill** — Runs a 72B teacher model to generate clinical responses for 2,500 patient scenarios
3. **Train** — Fine-tunes a 1.5B student model on the distilled data using LoRA adapters
4. **Compare** — Side-by-side web UI: your specialist vs. the base model

## Pipeline Scripts

| Script | What it does | GPU |
|---|---|---|
| `01_ingest_data.py` | Downloads datasets to a Modal Volume | None |
| `02_distill_teacher.py` | Generates clinical training data via Qwen 72B | A100-80GB |
| `03_train_student.py` | Fine-tunes Qwen 1.5B with LoRA (120 steps) | L4 |
| `04_test_model.py` | Runs a single query against your specialist | L4 |
| `05_compare_models.py` | FastAPI UI to compare specialist vs. base model | L4 |

## Setup

### Prerequisites
- [Modal account](https://modal.com) (free tier works)
- Hugging Face account with access to the datasets

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure secrets
Create a Modal secret named `hf-secret` with your Hugging Face token:
```bash
modal secret create hf-secret HF_TOKEN=hf_your_token_here
```

### 3. Create the Modal Volume
```bash
modal volume create mental-health-data
```

## Running the Pipeline

Run each step in order:

```bash
# Step 1 — Download datasets
modal run 01_ingest_data.py

# Step 2 — Generate training data via teacher model (~1-2 hours)
modal run 02_distill_teacher.py

# Step 3 — Fine-tune the specialist (~20 minutes)
modal run 03_train_student.py

# Step 4 — Test a single query
modal run 04_test_model.py

# Step 5 — Launch the comparison web UI
modal serve 05_compare_models.py
```

## Concepts Demonstrated

- **Knowledge distillation** — transferring expertise from a 72B to a 1.5B model
- **LoRA fine-tuning** — efficient parameter-efficient training with Unsloth
- **4-bit quantization** — running large models within GPU memory limits
- **Serverless ML** — Modal for on-demand GPU workloads
- **vLLM inference** — fast batched generation for the teacher step

## Datasets Used

- [ShenLab/MentalChat16K](https://huggingface.co/datasets/ShenLab/MentalChat16K)
- [Amod/mental_health_counseling_conversations](https://huggingface.co/datasets/Amod/mental_health_counseling_conversations)
- [UVSKKR/Ethical-Reasoning-in-Mental-Health-v1](https://huggingface.co/datasets/UVSKKR/Ethical-Reasoning-in-Mental-Health-v1)

## Note

This is a learning/demo project. The trained model is not a substitute for professional mental health support.

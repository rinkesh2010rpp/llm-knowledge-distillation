import modal
import os

image = modal.Image.debian_slim().pip_install(
    "datasets",
    "huggingface_hub",
    "pandas"
)

app = modal.App("mental-health-distill")
volume = modal.Volume.from_name("mental-health-data", create_if_missing=True)

DATASETS = [
    "ShenLab/MentalChat16K",
    "Amod/mental_health_counseling_conversations",
    "UVSKKR/Ethical-Reasoning-in-Mental-Health-v1"
]

# --- ADD THE SECRET HERE ---
@app.function(
    image=image, 
    volumes={"/data": volume}, 
    secrets=[modal.Secret.from_name("hf-secret")], # Use the secret we created
    timeout=600
)
def download_and_merge():
    from datasets import load_dataset
    from huggingface_hub import login
    
    # Use the token from the secret to log in
    token = os.environ["HF_TOKEN"]
    login(token=token)

    print("🚀 Starting data ingestion with authentication...")
    
    for ds_name in DATASETS:
        print(f"Downloading {ds_name}...")
        try:
            ds = load_dataset(ds_name, split="train")
            path = f"/data/{ds_name.replace('/', '_')}.jsonl"
            ds.to_json(path)
            print(f"✅ Saved {ds_name} to {path}")
        except Exception as e:
            print(f"❌ Failed to download {ds_name}: {e}")

    volume.commit()
    print("✅ Ingestion attempt finished.")

@app.local_entrypoint()
def main():
    download_and_merge.remote()
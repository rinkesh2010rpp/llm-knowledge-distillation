"""Download the base model to a Volume on a cheap CPU container.

The training GPU then loads it from the Volume instead of spending paid
H200 minutes on a ~70GB download.

    modal run 02_download_model.py
"""
import modal

MODEL = "unsloth/Qwen3.6-35B-A3B"

app = modal.App("sudarshana-distill-download")
volume = modal.Volume.from_name("sudarshana-distill", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub[hf_xet]")


@app.function(image=image, volumes={"/vol": volume}, cpu=4, memory=8192, timeout=3 * 3600)
def download():
    from huggingface_hub import snapshot_download

    path = snapshot_download(MODEL, local_dir=f"/vol/models/{MODEL.split('/')[-1]}")
    volume.commit()
    print(f"Downloaded {MODEL} to {path}")


@app.local_entrypoint()
def main():
    download.remote()

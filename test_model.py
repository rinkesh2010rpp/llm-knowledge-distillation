import modal

image = modal.Image.debian_slim(python_version="3.10").pip_install("unsloth", "torch", "transformers")
app = modal.App("mental-health-test")
volume = modal.Volume.from_name("mental-health-data")

@app.function(
    image=image,
    gpu="L4", 
    volumes={"/data": volume}
)
def test_specialist(user_query: str):
    from unsloth import FastLanguageModel
    import torch

    # Load your newly trained specialist
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "/data/mental_health_specialist_v1",
        max_seq_length = 2048,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)

    # Format the prompt exactly like the training data
    prompt = f"<|im_start|>user\n{user_query}<|im_end|>\n<|im_start|>assistant\nRationale:"

    inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens = 1000, use_cache = True)
    
    response = tokenizer.batch_decode(outputs)[0]
    parts = response.split("<|im_start|>assistant\n")
    assistant_text = parts[1].replace("<|im_end|>", "") if len(parts) > 1 else response
    print("\n" + "="*50)
    print("AI THERAPIST RESPONSE:")
    print(assistant_text)
    print("="*50 + "\n")

@app.local_entrypoint()
def main():
    # TEST CASE: Something complex and emotional
    query = "I've been working 80 hours a week and I feel like I'm a shell of a person. I don't enjoy anything anymore, and even seeing my friends feels like a chore."
    test_specialist.remote(query)
import html as html_lib
import modal
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

# 1. Environment Setup
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("unsloth", "torch", "transformers", "fastapi[standard]")
)
app = modal.App("mental-health-comparator")
volume = modal.Volume.from_name("mental-health-data")

# 2. Define the Inference Class
@app.cls(
    image=image,
    gpu="L4", # Uses 1 GPU to switch between models or run them sequentially
    volumes={"/data": volume},
    container_idle_timeout=300
)
class ModelComparator:
    @modal.enter()
    def load_models(self):
        from unsloth import FastLanguageModel
        import torch

        print("🚀 Loading both models (Base and Specialist)...")
        # Load Specialist (Your trained model)
        self.spec_model, self.spec_tokenizer = FastLanguageModel.from_pretrained(
            model_name="/data/mental_health_specialist_v1",
            max_seq_length=2048,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self.spec_model)

        # Load Base (Normal Qwen 1.5B)
        self.base_model, self.base_tokenizer = FastLanguageModel.from_pretrained(
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
            max_seq_length=2048,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self.base_model)

    @modal.method()
    def compare(self, query: str):
        # Helper to generate text
        def generate(model, tokenizer, text):
            prompt = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
            # Note: For specialist, we add 'Rationale:' to steer it
            if model == self.spec_model:
                prompt += "Rationale:"
            
            inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
            outputs = model.generate(**inputs, max_new_tokens=512, use_cache=True)
            decoded = tokenizer.batch_decode(outputs)[0]
            parts = decoded.split("<|im_start|>assistant\n")
            return parts[1].replace("<|im_end|>", "") if len(parts) > 1 else decoded

        return {
            "specialist": generate(self.spec_model, self.spec_tokenizer, query),
            "base": generate(self.base_model, self.base_tokenizer, query)
        }

# 3. FastAPI Web Interface
@app.function(image=image)
@modal.asgi_app()
def web_ui():
    web_app = FastAPI()

    @web_app.get("/", response_class=HTMLResponse)
    async def index():
        return """
        <html>
            <head>
                <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
                <style>
                    body { background: #f8f9fa; padding: 50px; }
                    .card { height: 100%; }
                    .response { white-space: pre-wrap; font-size: 0.9rem; }
                    .badge-spec { background-color: #0d6efd; }
                    .badge-base { background-color: #6c757d; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h2 class="mb-4 text-center">🧠 Clinical Specialist vs. Base Model</h2>
                    <form action="/compare" method="post" class="mb-5">
                        <div class="input-group">
                            <input type="text" name="query" class="form-control" placeholder="Describe a mental health concern..." required>
                            <button class="btn btn-primary" type="submit">Compare Side-by-Side</button>
                        </div>
                    </form>
                </div>
            </body>
        </html>
        """

    @web_app.post("/compare", response_class=HTMLResponse)
    async def run_comparison(request: Request):
        form = await request.form()
        query = form["query"]
        
        # Call the GPU function
        results = ModelComparator().compare.remote(query)

        safe_query = html_lib.escape(query)
        safe_specialist = html_lib.escape(results['specialist'])
        safe_base = html_lib.escape(results['base'])

        return f"""
        <html>
            <head><link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"></head>
            <body class="p-5 bg-light">
                <div class="container">
                    <a href="/" class="btn btn-outline-secondary mb-4">← Back</a>
                    <h4>Prompt: "{safe_query}"</h4>
                    <div class="row mt-4">
                        <div class="col-md-6">
                            <div class="card shadow">
                                <div class="card-header bg-primary text-white">Your Trained Specialist</div>
                                <div class="card-body response">{safe_specialist}</div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="card shadow">
                                <div class="card-header bg-secondary text-white">Base Qwen 1.5B (Normal)</div>
                                <div class="card-body response">{safe_base}</div>
                            </div>
                        </div>
                    </div>
                </div>
            </body>
        </html>
        """
    return web_app
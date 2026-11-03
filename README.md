# LLM Knowledge Distillation

Distilling large models into small, specialised ones. Training runs serverlessly on [Modal](https://modal.com).

| Project | What it does | Status |
|---|---|---|
| [`clinical/`](clinical/) | Qwen 72B → Qwen 1.5B mental-health specialist (single-turn Q&A). The first learning project. | Done |
| [`sudarshana/`](sudarshana/) | DeepSeek V4 Flash → small self-hosted model for the Sudarshana agent, trained on its logged tool-calling trajectories. | Paused: SFT beat the base model, GRPO added no clear gain ([results](sudarshana/README.md#results-paused-2026-10-15)) |

Each project is self-contained: its own numbered pipeline scripts, `requirements.txt` and README.

# LLM Knowledge Distillation

Distilling large models into small, specialised ones. Training runs serverlessly on [Modal](https://modal.com).

| Project | What it does | Status |
|---|---|---|
| [`clinical/`](clinical/) | Qwen 72B → Qwen 1.5B mental-health specialist (single-turn Q&A). The first learning project. | Done |
| [`sudarshana/`](sudarshana/) | DeepSeek V4 Flash → small self-hosted model for the Sudarshana agent, trained on its logged tool-calling trajectories. | In progress |

Each project is self-contained: its own numbered pipeline scripts, `requirements.txt` and README.

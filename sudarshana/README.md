# Sudarshana: distilling the agent into a small model

Sudarshana logs every model call (input, DeepSeek V4 Flash's reply, reasoning, token usage) to the `sudarshana-files` Modal Volume. This pipeline turns those logs into training data and trains a small model that can run the agent, self-hosted on Modal.

## Plan

| Step | Script | What it does |
|---|---|---|
| 1 | `01_build_dataset.py` | Logs → train / val / test JSONL (runs locally) |
| 2 | `02_download_model.py` | Base model (Qwen3.6-35B-A3B) → `sudarshana-distill` Volume, on a CPU container |
| 3 | `03_train_sft.py` | bf16 LoRA SFT on an H200, in resumable 2-hour slices |
| 4 | `04_validate.py` | Quick check on val: loss split by reasoning / reply / tool calls, base vs. trained, plus a few generations |
| 5 | `05_merge_adapter.py` | Merge a run's LoRA into the base model, for vLLM |
| 6 | `06_generate.py` | vLLM: each model makes every test decision itself |
| 7 | `07_judge.py` | LLM judge (Claude Haiku 5.5 via OpenRouter) compares each decision with DeepSeek's |
| 8 | `08_train_grpo.py` | GRPO with the judge as reward (work in progress, see below) |
| 9 | *(later)* | Serve with vLLM; shadow DeepSeek before switching |

## Step 1: build the dataset

```bash
pip install -r requirements.txt
```

Download the logs. On Windows, `modal volume get` fails if you name a destination folder, so run it from inside an empty one:

```bash
mkdir llm-calls && cd llm-calls && modal volume get sudarshana-files llm-calls . && cd ..
```

```bash
python 01_build_dataset.py --logs ./llm-calls/llm-calls --out ./data
```

Outputs in `data/` (git-ignored; it contains private conversations):

- `train.jsonl`, `val.jsonl`, `test.jsonl`: one line per logged call (main agent, subagent and summarizer alike, marked in `kind`). Each has `messages` (the full input, as sent), `target` (DeepSeek's reply, with `reasoning_content`), `tools`, and `n_tokens`.
- `split.json`: the frozen split. Delete it only if you mean to re-split.
- `stats.md`: counts, drops and token lengths.

### How the split works

- Split by **turn**, never by call: calls in one turn share almost all their context.
- **Test** = cron turns from 2026-10-10 and 10-11, plus 9 hand-picked Telegram turns (`TEST_TELEGRAM_TURNS`). It is frozen: new logs go to train, so test scores stay comparable across runs.
- **Val** = 5% of the remaining turns, at random, for early stopping only.
- Retries of the same Telegram message stay in the same split.

### Filtering and conversion

- Drops failed calls, outputs cut off at the max-token limit, calls to tools that don't exist, and empty outputs.
- Keeps 20% of the no-op "hold" cron turns in train; test keeps them all.
- Mid-conversation system messages ("Current time: …") become user messages, because Qwen's template rejects them.
- Nothing is truncated: the model trains on the same full context it will see when serving. Calls longer than `--max-tokens` (96K) are dropped. They all predate PR #41, after which Sudarshana summarises at ~70K DeepSeek tokens, so they don't reflect real use.

## Steps 2–3: train on Modal

On Windows Git Bash, set `MSYS_NO_PATHCONV=1`, or it rewrites `/data/...` into a Windows path inside the Volume.

```bash
modal run 02_download_model.py
```

```bash
MSYS_NO_PATHCONV=1 modal volume put sudarshana-distill data/train.jsonl /data/train.jsonl
```

```bash
MSYS_NO_PATHCONV=1 modal volume put sudarshana-distill data/val.jsonl /data/val.jsonl
```

```bash
modal run --detach 03_train_sft.py --run-name run1
```

Each run:

- trains for at most **2 hours**, then saves and measures loss on 30 val examples. Modal's timeout (2h15m) is the hard cap, about $10 on an H200;
- trains bf16 LoRA (Unsloth advises against 4-bit QLoRA for Qwen3.5/3.6) at up to 96K tokens, with loss only on DeepSeek's reply;
- goes through the examples in a fixed order (longest first, so a memory problem shows up in the first minute) and saves a checkpoint every 15 minutes;
- uses a constant learning rate after warmup, so slices can be chained.

Rerun the same command to continue from where the last run stopped. Progress is in `/vol/runs/<run-name>/log.jsonl` and `val.jsonl`.

## Steps 4–7: evaluate

```bash
modal run --detach 04_validate.py --run-name run1
```

```bash
modal run --detach 05_merge_adapter.py --run-name run1
```

```bash
MSYS_NO_PATHCONV=1 modal volume put sudarshana-distill data/test.jsonl /data/test.jsonl
```

```bash
modal run --detach 06_generate.py --model run1 --split test
```

```bash
modal run 07_judge.py --action run --model run1 --split test
```

`06_generate.py` takes `--model base` for the untrained model. The judge needs `OPENROUTER_API_KEY` in a `.env` in this folder (git-ignored; keep only that key in it, since Modal uploads the whole file).

How the judge works:

- It sees the agent's full system prompt, the message that started the turn, the conversation so far (very long tool results trimmed), and two candidate next steps: DeepSeek's and the model's.
- Both answers are shown the same way (the action and any message, no reasoning), identical answers are scored a tie without asking, and every pair is judged twice with A and B swapped.
- `--action submit` / `--action collect` use OpenRouter's Batch API instead: half the price, results within 24 hours.

Choosing the judge: on 15 decisions graded blind as a reference, Claude Haiku 5.5 agreed 11/15 (and 73–80% with itself when A/B were swapped). Jev, TypeSafe's decision model, agreed 5/15 and only 47% with itself, so it was dropped. Avoid DeepSeek as the judge (it would be grading its own answers) and Qwen (the student's own family).

## Step 8: GRPO (work in progress)

Test results so far (489 held-out decisions, Haiku judge; win / tie / loss against DeepSeek, then net = win − loss):

| Model | Win | Tie | Loss | Net |
|---|---|---|---|---|
| Base Qwen3.6-35B-A3B | 14.7% | 41.1% | 44.2% | −30 |
| Run 1 (SFT) | 15.7% | 47.4% | 37.0% | −21 |

SFT turned losses into ties; it did not add wins. GRPO is the next lever: the model writes several answers per situation, the judge scores each against DeepSeek's, and the better ones are reinforced.

`08_train_grpo.py` uses Unsloth's `fast_inference=True`. vLLM and the trainer share one copy of the weights, including the MoE experts with LoRA on them (Unsloth ≥ 2026.10.3), so it needs one GPU and no weight syncing. The reward comes from `judge_core.py`, the same code `07_judge.py` evaluates with. It is graded by the judge's 1–5 score gap rather than win/tie/loss, so a group where every answer loses still has a signal.

What it took to get running (every failure was out of memory):

- **Root cause:** Unsloth's autotuner batched several 40–94K-token answers into one no-grad log-prob pass. It sized the batch with an estimate that ignores the model's own activations, which caused the recurring 11.45 GB allocation failure. On a B300 the same failure surfaced as "illegal memory access", with 18 MB free. The fix is `unsloth_grpo_mini_batch = num_generations`, one answer per pass.
- **GPU:** B200. An H200 is too small at 94K: vLLM needs ~85 GB besides its cache and training ~60 GB. A B300 (sm_103) is not in Unsloth's tuned MoE kernel list and used ~3× the memory.
- **vLLM:** a fixed 55% share with standby off. Unsloth disables standby for multimodal checkpoints, and forcing it on didn't free memory.
- **Gotchas:** `finetune_vision_layers=False`; install vLLM before Unsloth; skip situations where DeepSeek's answer is longer than the answer cap, or long-but-correct answers get scored 0.
- `--diagnose` copies the full console to the Volume (Modal's logs expire) and records GPU memory at every phase.

The B200 test (8 answers per situation, 23 steps): median ~96 s per step, peak 141.5 GB, held-out reward flat at 0.48 ± 0.02 at lr 5e-6. The defaults are now the first real run: 150 steps, 4 answers, lr 1e-5, β = 0, answers up to 8K tokens, a held-out check every 25 steps.

```bash
modal run --detach 08_train_grpo.py --run-name grpo-run1
```

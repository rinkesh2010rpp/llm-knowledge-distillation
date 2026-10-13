# Sudarshana: distilling the agent into a small model

Sudarshana logs every model call (input, DeepSeek V4 Flash's reply, reasoning, token usage) to the `sudarshana-files` Modal Volume. This pipeline turns those logs into training data and trains a small model that can run the agent, self-hosted on Modal.

## Plan

| Step | Script | What it does |
|---|---|---|
| 1 | `01_build_dataset.py` | Logs → train / val / test JSONL (runs locally) |
| 2 | `02_download_model.py` | Base model (Qwen3.6-35B-A3B) → `sudarshana-distill` Volume, on a CPU container |
| 3 | `03_train_sft.py` | bf16 LoRA SFT on an H200, in resumable 2-hour slices |
| 4 | *(next)* | Judge: score the trained model against DeepSeek on test |
| 5 | *(if needed)* | Single-step RL (GRPO) with the judge as reward |
| 6 | *(later)* | Serve with vLLM; shadow DeepSeek before switching |

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

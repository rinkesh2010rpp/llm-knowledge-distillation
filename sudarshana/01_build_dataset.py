"""Turn Sudarshana's logged model calls into train / val / test SFT data.

Input: a local copy of the `llm-calls` folder from the `sudarshana-files` Volume
(see README). Every logged call becomes one example, exactly as it happened:
the full input (system prompt, history, tools) and DeepSeek's reply as target.
Main-agent, subagent and summarizer calls are all included.

    python 01_build_dataset.py --logs ./llm-calls --out ./data

The test/val split is frozen in split.json on first run. Later runs reuse it,
and turns that weren't seen before go to train, so test scores stay comparable.
"""
import argparse
import collections
import glob
import gzip
import json
import os
import random
import re

# Telegram turns hand-picked for test: a spread of request types. Cron turns
# from TEST_DATES are added automatically.
TEST_TELEGRAM_TURNS = {
    "bb39f06e3dec",  # what's waiting on me? (status)
    "3d31fe67ad0b",  # curl + summarise pages, one flagged as injection
    "77ea8d66654a",  # pushback on a proposal
    "3500ce5fcedc",  # "I didn't understand the issue nor the fix"
    "df7ec98f9501",  # PR merged, research other agents
    "442ba55cd2ad",  # user's own math proof idea
    "6db94e2431c2",  # summarise today
    "df06fc585a6a",  # technical directive on a build cache
    "009c37a2931a",  # bring me up to speed
}
TEST_DATES = {"2026-10-10", "2026-10-11"}
VAL_FRACTION = 0.05
HOLD_KEEP = 0.2  # fraction of no-op "hold" cron turns kept in train
SEED = 0

SUMMARIZER_PREFIX = "<role> Context Extraction Assistant </role>"
RESUME_PREFIX = "You are in the middle of a conversation that has been summarized"
HOLD_RE = re.compile(r"\bhold\b|no-op", re.I)


def text_of(content):
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, dict):
            parts.append(block.get("text") or "")
        else:
            parts.append(str(block))
    return "".join(parts)


def load_turns(logs_dir):
    turns = collections.defaultdict(list)
    for path in sorted(glob.glob(os.path.join(logs_dir, "20*", "*.jsonl.gz"))):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                turns[rec["turn_id"]].append(rec)
    for recs in turns.values():
        recs.sort(key=lambda r: int(r["seq"]))
    return turns


def load_tools(logs_dir):
    tools = {}
    for path in glob.glob(os.path.join(logs_dir, "tools", "*.json")):
        with open(path, encoding="utf-8") as fh:
            tools[os.path.splitext(os.path.basename(path))[0]] = json.load(fh)
    return tools


def user_message(recs):
    """The message that started the turn (ignores summarizer plumbing)."""
    for rec in recs:
        humans = [text_of(m["data"]["content"]) for m in rec["input"] if m["type"] == "human"]
        humans = [h for h in humans if not h.startswith((SUMMARIZER_PREFIX, RESUME_PREFIX))]
        if humans:
            return humans[-1]
    return ""


def is_hold(recs):
    if recs[0]["trigger"] == "telegram" or len(recs) > 5:
        return False
    last = recs[-1].get("output")
    return bool(last) and bool(HOLD_RE.search(text_of(last["data"].get("content"))))


def make_split(turns, path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            split = json.load(fh)
    else:
        # Retries of one Telegram message share a split, so group by the message.
        groups = collections.defaultdict(list)
        for tid, recs in turns.items():
            key = ("tg", user_message(recs)) if recs[0]["trigger"] == "telegram" else ("cron", tid)
            groups[key].append(tid)
        test, rest = [], []
        for tids in groups.values():
            recs = [r for t in tids for r in turns[t]]
            date = min(r["ts"] for r in recs)[:10]
            is_test = any(t in TEST_TELEGRAM_TURNS for t in tids) or (
                recs[0]["trigger"] != "telegram" and date in TEST_DATES
            )
            (test if is_test else rest).append(sorted(tids))
        rng = random.Random(SEED)
        rest.sort()
        rng.shuffle(rest)
        n_val = round(len(rest) * VAL_FRACTION)
        split = {
            "test": sorted(t for g in test for t in g),
            "val": sorted(t for g in rest[:n_val] for t in g),
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(split, fh, indent=1)
    lookup = {t: "test" for t in split["test"]}
    lookup.update({t: "val" for t in split["val"]})
    return lookup


def convert(msg, first):
    kind, data = msg["type"], msg["data"]
    if kind == "system":
        # Qwen's template only allows a system message at the start; Sudarshana
        # injects "Current time:" mid-thread, so those become user messages.
        content = text_of(data["content"])
        return {"role": "system", "content": content} if first else {"role": "user", "content": content}
    if kind == "human":
        return {"role": "user", "content": text_of(data["content"])}
    if kind == "ai":
        out = {"role": "assistant", "content": text_of(data.get("content"))}
        reasoning = data.get("additional_kwargs", {}).get("reasoning_content")
        if reasoning:
            out["reasoning_content"] = reasoning
        if data.get("tool_calls"):
            out["tool_calls"] = [
                {"id": tc.get("id"), "type": "function", "function": {"name": tc["name"], "arguments": tc["args"]}}
                for tc in data["tool_calls"]
            ]
        return out
    if kind == "tool":
        return {"role": "tool", "content": text_of(data.get("content")), "tool_call_id": data.get("tool_call_id")}
    raise ValueError(f"unknown message type {kind}")


def kind_of(rec):
    if not rec.get("tools_hash"):
        return "summarizer"
    system = text_of(rec["input"][0]["data"]["content"]) if rec["input"] else ""
    return "main" if system.startswith("You are Sudarshana") else "subagent"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True, help="local copy of the llm-calls folder")
    ap.add_argument("--out", default="data")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.6-35B-A3B")
    # Real inputs top out near this since PR #41 (summarization at ~70K
    # DeepSeek tokens). Longer calls predate it and don't reflect real use.
    ap.add_argument("--max-tokens", type=int, default=98304)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    os.makedirs(args.out, exist_ok=True)
    turns = load_turns(args.logs)
    tools_by_hash = load_tools(args.logs)
    split_of = make_split(turns, os.path.join(args.out, "split.json"))
    rng = random.Random(SEED)

    stats = collections.Counter()
    lengths = collections.defaultdict(list)
    files = {}

    for tid, recs in sorted(turns.items()):
        split = split_of.get(tid, "train")
        hold = is_hold(recs)
        if hold and split == "train" and rng.random() > HOLD_KEEP:
            stats["drop: hold turn (down-weighted)"] += len(recs)
            continue
        for rec in recs:
            out = rec.get("output")
            if not out:
                stats["drop: failed call"] += 1
                continue
            data = out["data"]
            if data.get("response_metadata", {}).get("finish_reason") == "length":
                stats["drop: hit max output tokens"] += 1
                continue
            tools = tools_by_hash.get(rec["tools_hash"]) if rec.get("tools_hash") else None
            if rec.get("tools_hash") and tools is None:
                stats["drop: tool schema missing"] += 1
                continue
            names = {t["function"]["name"] for t in tools or []}
            if any(tc["name"] not in names for tc in data.get("tool_calls", [])):
                stats["drop: unknown tool name"] += 1
                continue
            if not data.get("tool_calls") and not text_of(data.get("content")).strip():
                stats["drop: empty output"] += 1
                continue

            messages = [convert(m, first=(i == 0)) for i, m in enumerate(rec["input"])]
            target = convert(out, first=False)
            rendered = tok.apply_chat_template(messages + [target], tools=tools, tokenize=False)
            n_tokens = len(tok.encode(rendered, add_special_tokens=False))
            if n_tokens > args.max_tokens:
                stats["drop: longer than max-tokens (pre-PR #41)"] += 1
                continue

            kind = kind_of(rec)
            if split not in files:
                files[split] = open(os.path.join(args.out, f"{split}.jsonl"), "w", encoding="utf-8")
            files[split].write(json.dumps({
                "id": f"{tid}-{rec['seq']}", "turn_id": tid, "split": split, "kind": kind,
                "trigger": rec["trigger"], "ts": rec["ts"], "hold": hold, "n_tokens": n_tokens,
                "tools": tools, "messages": messages, "target": target,
            }, ensure_ascii=False) + "\n")
            stats[f"keep: {split}"] += 1
            stats[f"kind: {kind}"] += 1
            lengths[split].append(n_tokens)

    for fh in files.values():
        fh.close()
    report = [f"# Dataset stats (max_tokens={args.max_tokens}, tokenizer={args.tokenizer})", ""]
    report += [f"- {k}: {v}" for k, v in sorted(stats.items())]
    report += ["", "| split | examples | median tokens | p90 | max | total tokens |", "|---|---|---|---|---|---|"]
    for name, ls in sorted(lengths.items()):
        ls = sorted(ls)
        report.append(f"| {name} | {len(ls)} | {ls[len(ls) // 2]:,} | {ls[int(len(ls) * 0.9)]:,} | {ls[-1]:,} | {sum(ls):,} |")
    text = "\n".join(report) + "\n"
    with open(os.path.join(args.out, "stats.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()

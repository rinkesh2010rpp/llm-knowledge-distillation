"""Score generated decisions against DeepSeek's with an LLM judge (OpenRouter Batch API).

For each example, the judge (Claude Haiku 5.5 by default) sees the situation
the agent was in (its full system prompt, the message that started the turn,
the history) and two candidate next steps: DeepSeek's and the model's. It
writes a short analysis, then picks the better step and scores both.

To keep verdicts about the decision, not presentation:
- both answers are shown the same way: the action and any message, no
  reasoning, arguments as plain text;
- identical answers are scored a tie without asking;
- every pair is judged twice, A and B swapped, and the scores averaged.

Normal requests, in parallel (minutes):

    modal run 07_judge.py --action run --model run1 --split test

Or the Batch API at half price, results within 24h, in two steps (`collect`
reports progress until every batch is done):

    modal run 07_judge.py --action submit --model run1 --split test
    modal run 07_judge.py --action collect --model run1 --split test

Output: /vol/eval/<split>/<model>.<judge>.judged.jsonl and .summary.json.
Needs OPENROUTER_API_KEY in a `.env` next to this file (git-ignored; only
that key, since Modal uploads the whole file).
"""
import os

import modal

from judge_core import (BATCH_URL, JUDGE_MODEL, MESSAGES_URL, REQUESTS_PER_BATCH, api, decision,
                        judge_request, load, parse_verdict)

app = modal.App("sudarshana-distill-judge")
volume = modal.Volume.from_name("sudarshana-distill", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").add_local_python_source("judge_core")
secrets = [modal.Secret.from_dotenv(os.path.dirname(os.path.abspath(__file__)))]

def build_requests(model, split, limit):
    examples, gens = load(model, split)
    if limit:
        gens = gens[:limit]
    requests = []
    for gen in gens:
        row, mine, deepseek = decision(examples[gen["id"]], gen)
        if row["identical"]:
            continue
        ex = examples[gen["id"]]
        requests.append(judge_request(f"{gen['id']}__model_A", ex, mine, deepseek))
        requests.append(judge_request(f"{gen['id']}__model_B", ex, deepseek, mine))
    return requests


@app.function(image=image, volumes={"/vol": volume}, secrets=secrets, timeout=3600)
def run(model: str, split: str, limit: int, judge_model: str, workers: int):
    """Normal (non-batch) requests in parallel: minutes instead of hours, 2x the price."""
    import time
    from concurrent.futures import ThreadPoolExecutor

    api_key = os.environ["OPENROUTER_API_KEY"].strip()
    requests = build_requests(model, split, limit)
    usage = {"input_tokens": 0, "output_tokens": 0}

    def one(req):
        for attempt in range(4):
            try:
                body = api("POST", MESSAGES_URL, api_key, {"model": judge_model, **req["body"]})
                for k in usage:
                    usage[k] += (body.get("usage") or {}).get(k, 0)
                verdict = parse_verdict(body)
                if verdict is None:
                    raw = "".join(b.get("text", "") for b in body.get("content", []))
                    print(f"[run] {req['custom_id']}: unparseable (stop={body.get('stop_reason')}): {raw[:600]!r}")
                return req["custom_id"], verdict
            except RuntimeError as e:
                if attempt == 3 or not any(f"HTTP {c}" in str(e) for c in (408, 429, 500, 502, 503, 504, 529)):
                    print(f"[run] {req['custom_id']}: {str(e)[:300]}")
                    return req["custom_id"], None
                time.sleep(5 * 2 ** attempt)

    t0 = time.time()
    with ThreadPoolExecutor(workers) as pool:
        results = dict(pool.map(one, requests))
    print(f"[run] {len(requests)} judge calls in {time.time() - t0:.0f}s; tokens in {usage['input_tokens']:,}, out {usage['output_tokens']:,}")
    score(model, split, limit, results, judge_model, tag="haiku" if judge_model == JUDGE_MODEL else judge_model.split("/")[-1])


@app.function(image=image, volumes={"/vol": volume}, secrets=secrets, timeout=3600)
def submit(model: str, split: str, limit: int, judge_model: str):
    import json

    api_key = os.environ["OPENROUTER_API_KEY"].strip()
    requests = build_requests(model, split, limit)
    batch_ids = []
    for i in range(0, len(requests), REQUESTS_PER_BATCH):
        # endpoint and model must come before requests in the body (stream-parsed).
        resp = api("POST", BATCH_URL, api_key, {"endpoint": "/v1/messages", "model": judge_model, "requests": requests[i:i + REQUESTS_PER_BATCH]})
        batch_ids.append(resp["id"])
        print(f"[submit] batch {resp['id']}: {len(requests[i:i + REQUESTS_PER_BATCH])} requests, status {resp['status']}")
    state = {"model": model, "split": split, "limit": limit, "judge_model": judge_model, "batch_ids": batch_ids, "requests": len(requests)}
    with open(f"/vol/eval/{split}/{model}.batches.json", "w") as fh:
        json.dump(state, fh, indent=1)
    volume.commit()
    print(f"[submit] {len(requests)} judge requests in {len(batch_ids)} batches")


@app.function(image=image, volumes={"/vol": volume}, secrets=secrets, timeout=3600)
def collect(model: str, split: str):
    import json

    api_key = os.environ["OPENROUTER_API_KEY"].strip()
    with open(f"/vol/eval/{split}/{model}.batches.json") as fh:
        state = json.load(fh)
    results, pending = {}, []
    for bid in state["batch_ids"]:
        batch = api("GET", f"{BATCH_URL}/{bid}", api_key)
        counts = batch.get("request_counts") or {}
        print(f"[collect] {bid}: {batch['status']} ({counts.get('completed', 0)}/{counts.get('total', '?')} done, {counts.get('failed', 0)} failed)")
        if batch["status"] in ("validating", "in_progress", "finalizing"):
            pending.append(bid)
            continue
        if batch.get("error"):
            print(f"[collect] {bid} error: {batch['error']}")
        for item in batch.get("results") or []:
            resp = item.get("response") or {}
            results[item["custom_id"]] = parse_verdict(resp.get("body") or {}) if resp.get("status_code") == 200 else None
    if pending:
        print(f"[collect] {len(pending)} batches still running; run collect again later")
        return
    score(model, split, state["limit"], results, state["judge_model"], tag="haiku-batch")


def score(model, split, limit, results, judge_model, tag):
    """Turn verdicts into per-decision rows and a summary, written next to the generations."""
    import collections
    import json

    examples, gens = load(model, split)
    if limit:
        gens = gens[:limit]
    rows = []
    for gen in gens:
        row, _, _ = decision(examples[gen["id"]], gen)
        if row["identical"]:
            rows.append({**row, "comparison": "tie", "p_model_better": 0.0, "p_tie": 1.0, "p_deepseek_better": 0.0,
                         "model_score": None, "deepseek_score": None, "swap_agrees": True})
            continue
        first, second = results.get(f"{gen['id']}__model_A"), results.get(f"{gen['id']}__model_B")
        if not first or not second:
            rows.append({**row, "error": "missing or unparseable verdict"})
            continue
        # Map each verdict to model / deepseek / tie, then average the two orders.
        v1 = {"A": "model", "B": "deepseek", "tie": "tie"}[first["winner"]]
        v2 = {"A": "deepseek", "B": "model", "tie": "tie"}[second["winner"]]
        votes = collections.Counter([v1, v2])
        p = {k: votes[k] / 2 for k in ("model", "tie", "deepseek")}
        rows.append({
            **row,
            "comparison": "model_better" if p["model"] > max(p["deepseek"], p["tie"]) else "deepseek_better" if p["deepseek"] > max(p["model"], p["tie"]) else "tie",
            "p_model_better": p["model"], "p_tie": p["tie"], "p_deepseek_better": p["deepseek"],
            "swap_agrees": v1 == v2,
            "model_score": (first["score_a"] + second["score_b"]) / 2,
            "deepseek_score": (first["score_b"] + second["score_a"]) / 2,
            "analysis_model_A": first.get("analysis"), "analysis_model_B": second.get("analysis"),
        })

    out_dir = f"/vol/eval/{split}"
    with open(f"{out_dir}/{model}.{tag}.judged.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    def summarize(rs):
        ok = [r for r in rs if "comparison" in r]
        if not ok:
            return {"n": len(rs), "errors": len(rs)}
        count = collections.Counter(r["comparison"] for r in ok)
        asked = [r for r in ok if not r["identical"]]
        scored = [r for r in asked if r["model_score"] is not None]
        tool_rows = [r for r in rs if r["same_first_tool"] is not None]
        return {
            "n": len(rs), "judged": len(ok), "errors": len(rs) - len(ok), "identical": sum(r["identical"] for r in rs),
            "model_better": round(count["model_better"] / len(ok), 3), "tie": round(count["tie"] / len(ok), 3),
            "deepseek_better": round(count["deepseek_better"] / len(ok), 3),
            "as_good_or_better": round((count["model_better"] + count["tie"]) / len(ok), 3),
            "mean_model_score": round(sum(r["model_score"] for r in scored) / len(scored), 2) if scored else None,
            "mean_deepseek_score": round(sum(r["deepseek_score"] for r in scored) / len(scored), 2) if scored else None,
            "swap_agreement": round(sum(r["swap_agrees"] for r in asked) / len(asked), 3) if asked else None,
            "valid": round(sum(r["valid"] for r in rs) / len(rs), 3),
            "same_action_type": round(sum(r["same_action_type"] for r in rs) / len(rs), 3),
            "same_first_tool": round(sum(r["same_first_tool"] for r in tool_rows) / len(tool_rows), 3) if tool_rows else None,
        }

    summary = {"model": model, "split": split, "judge": judge_model, "all": summarize(rows)}
    for key in ("kind", "trigger"):
        for value in sorted({r[key] for r in rows}):
            summary[f"{key}={value}"] = summarize([r for r in rows if r[key] == value])
    with open(f"{out_dir}/{model}.{tag}.summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    volume.commit()
    print(json.dumps(summary["all"], indent=1))


@app.local_entrypoint()
def main(action: str = "run", model: str = "run1", split: str = "test", limit: int = 0, judge_model: str = JUDGE_MODEL, workers: int = 16):
    if action == "run":
        run.remote(model, split, limit, judge_model, workers)
    elif action == "submit":
        submit.remote(model, split, limit, judge_model)
    elif action == "collect":
        collect.remote(model, split)
    else:
        raise SystemExit("--action must be run, submit or collect")

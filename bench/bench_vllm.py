#!/usr/bin/env python3
import argparse
import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PROMPT = (
    "You are a careful technical writer. Explain, step by step and in plain "
    "language, how a GPU executes the decode phase of a transformer language "
    "model: what is read from memory for every generated token, why the key "
    "and value cache grows with context length, and why batching many "
    "requests together raises total throughput. Use concrete numbers."
)


def completions_url(base_url):
    base = base_url.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    return base + "/completions"


def stream_one(url, model, max_tokens, timeout):
    body = json.dumps({
        "model": model,
        "prompt": PROMPT,
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})

    start = time.perf_counter()
    first = last = None
    chunks = 0
    usage_tokens = None

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if event.get("usage"):
                usage_tokens = event["usage"].get("completion_tokens")
            choices = event.get("choices") or []
            if choices and choices[0].get("text"):
                now = time.perf_counter()
                if first is None:
                    first = now
                last = now
                chunks += 1

    if first is None:
        raise RuntimeError("server returned no tokens")

    return {
        "start": start,
        "first": first,
        "last": last,
        "tokens": usage_tokens or chunks,
    }


def decode_rate(result):
    span = result["last"] - result["first"]
    if result["tokens"] < 2 or span <= 0:
        return 0.0
    return (result["tokens"] - 1) / span


def run_batch(url, model, max_tokens, concurrency, timeout):
    gate = threading.Barrier(concurrency)

    def worker(_):
        gate.wait()
        return stream_one(url, model, max_tokens, timeout)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(worker, range(concurrency)))

    start = min(r["start"] for r in results)
    end = max(r["last"] for r in results)
    total = sum(r["tokens"] for r in results)
    return total / (end - start), results


def parse_levels(text):
    try:
        levels = [int(x) for x in text.split(",") if x.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError("expected a comma separated list like 1,8,32")
    if not levels or min(levels) < 1:
        raise argparse.ArgumentTypeError("concurrency levels must be positive")
    return levels


def main():
    ap = argparse.ArgumentParser(description="Measure vLLM decode throughput for decode-roofline.")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--concurrency", type=parse_levels, default=[1, 8, 32])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--out", default="measured.json")
    args = ap.parse_args()

    url = completions_url(args.base_url)

    try:
        print("warmup...", file=sys.stderr)
        stream_one(url, args.model, args.max_tokens, args.timeout)

        decode, ttft = [], []
        for i in range(args.repeats):
            r = stream_one(url, args.model, args.max_tokens, args.timeout)
            decode.append(decode_rate(r))
            ttft.append((r["first"] - r["start"]) * 1000)
            print(f"single stream run {i + 1}: {decode[-1]:.1f} tok/s decode", file=sys.stderr)

        levels = []
        for c in args.concurrency:
            runs = []
            for i in range(args.repeats):
                tps, _ = run_batch(url, args.model, args.max_tokens, c, args.timeout)
                runs.append(tps)
                print(f"concurrency {c} run {i + 1}: {tps:.1f} tok/s total", file=sys.stderr)
            levels.append({"concurrency": c, "output_tokens_per_sec": round(statistics.median(runs), 2)})
    except urllib.error.HTTPError as e:
        sys.exit(f"server error {e.code}: {e.read().decode(errors='replace')}")
    except urllib.error.URLError as e:
        sys.exit(f"cannot reach {url}: {e.reason}")

    out = {
        "model": args.model,
        "base_url": args.base_url,
        "max_tokens": args.max_tokens,
        "repeats": args.repeats,
        "single_stream": {
            "decode_tokens_per_sec": round(statistics.median(decode), 2),
            "ttft_ms": round(statistics.median(ttft), 2),
        },
        "concurrency": levels,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

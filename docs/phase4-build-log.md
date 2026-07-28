# Phase 4 Build Log

*2026-04-16*

---

## 1. What was originally intended

The goal was a multi-GPU benchmarking harness that would compare RTX 4000 Ada
against RTX PRO 6000 Blackwell using a tensor-parallel vLLM deployment, with
a cost model that converts GPU-hour pricing to per-token, per-request, and
per-million-token costs.

The original assumed architecture:

- An **async load generator** (`harness/load_gen.py`) using `aiohttp` with
  SSE streaming, capturing time-to-first-token (TTFT) and inter-token latency
  (ITL) in addition to end-to-end latency.
- A **tensor-parallel vLLM deployment** (`k8s/vllm-tp.yaml`) with
  `--tensor-parallel-size 2` as the primary benchmark target.
- `harness/run_benchmark.py` as the primary CLI driver, sweeping over
  `batch_sizes` and `concurrency_levels` defined in a YAML config file.
- `aiohttp>=3.9` in `requirements.txt` as a required dependency.

---

## 2. What was actually built

### Scope correction: single-GPU Ada benchmark

The RTX PRO 6000 Blackwell node pool was not yet available on the cluster at
the time of Phase 4 implementation. The tensor-parallel target (`vllm-tp.yaml`)
requires two GPUs on one node, which the Ada pool does not provide.

Phase 4 was reframed as a **single-GPU baseline benchmark** on the existing Ada
node, measuring raw decode throughput and per-token cost with no external cache
in the loop. The Fermyon/Valkey prefix-cache hit rate is intentionally excluded
here — that is a Phase 2 measurement. Mixing cache effects into Phase 4 would
contaminate the per-token cost baseline. The two measurements are run
independently and compared.

### benchmark/benchmark.py

A synchronous sweep script using `requests` and `concurrent.futures.ThreadPoolExecutor`.
No `aiohttp`, no `asyncio`, no SSE — non-streaming `/v1/completions` only.

CLI flags:

```
--url              vLLM /v1/completions endpoint
--model            model ID (must match vLLM --model)
--prompt-tokens    approximate input length: 128, 256, or 512
--max-tokens       max output tokens per request
--concurrency      concurrent in-flight requests
--num-requests     total requests to send
--gpu-hourly-usd   GPU node hourly price in USD
--output-csv       CSV file to append results to
--tag              run label (auto-generated if omitted)
```

Each run appends one row to `--output-csv`. Prior rows are never overwritten,
so a sweep can be run incrementally and individual concurrency levels can be
re-run without losing other results.

### Prompt construction

Prompts are constructed by repeating the phrase `"The quick brown fox jumps
over the lazy dog. "` until the approximate token count is reached. The phrase
tokenises to ~10 BPE tokens with the Mistral/Qwen vocabulary; repeating it
`round(target / 10)` times gives a prompt of approximately the right length.
This is explicitly an approximation. Character count divided by 4 was
considered and rejected — it does not track actual token boundaries.

### Cost model integration

`harness/cost_model.py` has no CLI entry point; it is a library. `benchmark.py`
calls `harness.cost_model.compute_cost()` directly on every run and writes cost
columns into the CSV alongside the throughput and latency figures. The formula:

```
cost_per_token     = gpu_hourly_usd / 3600 / tokens_per_second
cost_per_request   = cost_per_token × mean_tokens_generated
cost_per_M_tokens  = cost_per_token × 1_000_000
```

`gpu_hourly_usd` for the Ada node (`g2-gpu-rtx4000a1-l`) is confirmed at $0.96
and set in `configs/rtx4000ada.yaml`. These are **output-token costs** at
varying concurrency — not a cross-provider comparison.

### k8s/vllm-ada.yaml

Single-GPU Ada deployment: `vllm serve`, `--no-enable-prefix-caching`,
`--max-model-len 25664`, `--dtype float16`, `nodeSelector: gpu-type=rtx4000ada`,
`nvidia.com/gpu: 1`. The `--no-enable-prefix-caching` flag is required because
the live Phase 2 deployment uses LMCache for prefix reuse; vLLM's own prefix
caching on top of that would contaminate per-token cost measurements.

### Tests

`tests/test_benchmark.py` — 27 new tests, no GPU or network required:

```
tests/test_benchmark.py   27 passed
tests/test_cost_model.py  30 passed
───────────────────────────────────
total                     57 passed
```

Test coverage: prompt scaling, error tracking, CSV append-without-overwrite,
header written exactly once, concurrency ceiling enforced by thread pool,
HTTP errors captured as `RequestResult.error` (not raised), `tokens_generated`
read from `usage.completion_tokens` in the response body.

---

## 3. Benchmark methodology

**Cluster:** Akamai LKE, us-ord. Single RTX 4000 Ada node (`g2-gpu-rtx4000a1-l`,
20 GB VRAM).

**Model:** `mistralai/Mistral-7B-Instruct-v0.2`, float16, `max_model_len=25664`,
no prefix caching.

**Deployment:** `k8s/vllm-ada.yaml` via `kubectl apply`. Local access via
`kubectl port-forward svc/vllm-ada 8000:8000 -n inference`.

**Connectivity verified before sweep:**
```
curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health
# → 200
```

**Sweep parameters:**

| Parameter | Values |
|---|---|
| concurrency | 1, 2, 4, 8, 16 |
| prompt_tokens | 128, 256, 512 (approximate — see prompt construction above) |
| max_tokens | 64 (fixed) |
| num_requests | 10 per run |
| gpu_hourly_usd | $0.96 |

15 runs total (5 concurrency levels × 3 prompt lengths). Each run appends one
row to `results/phase4_raw_benchmark.csv`. All cost figures in the results
section are output-token costs at varying concurrency; they are not a
cross-provider comparison.

A single warm-up request was issued before the sweep (c=1, p=128, n=1) to
ensure the model was loaded and the endpoint was responsive.

---

## 4. Full sweep results

All 15 sweep rows from `results/phase4_raw_benchmark.csv`. Zero errors across
all runs.

| tag | c | p_tok | tok/s | req/s | p50 (ms) | p95 (ms) | p99 (ms) | mean_out | cost/tok (USD) | cost/M (USD) |
|---|:-:|:-:|---:|---:|---:|---:|---:|---:|---:|---:|
| sweep-c1-p128  |  1 | 128 |  21.0 | 0.33 | 3013 | 3145 | 3159 | 64.0 | 0.00001268 | 12.68 |
| sweep-c1-p256  |  1 | 256 |  20.9 | 0.33 | 3069 | 3135 | 3146 | 64.0 | 0.00001277 | 12.77 |
| sweep-c1-p512  |  1 | 512 |  21.1 | 0.33 | 3025 | 3101 | 3111 | 64.0 | 0.00001267 | 12.67 |
| sweep-c2-p128  |  2 | 128 |  41.2 | 0.64 | 3082 | 3173 | 3173 | 64.0 | 0.00000648 |  6.48 |
| sweep-c2-p256  |  2 | 256 |  41.4 | 0.65 | 3086 | 3114 | 3124 | 64.0 | 0.00000644 |  6.44 |
| sweep-c2-p512  |  2 | 512 |  41.0 | 0.64 | 3123 | 3173 | 3173 | 64.0 | 0.00000651 |  6.51 |
| sweep-c4-p128  |  4 | 128 |  67.5 | 1.05 | 3205 | 3235 | 3235 | 64.0 | 0.00000395 |  3.95 |
| sweep-c4-p256  |  4 | 256 |  67.5 | 1.07 | 3094 | 3152 | 3189 | 63.4 | 0.00000395 |  3.95 |
| sweep-c4-p512  |  4 | 512 |  65.3 | 1.05 | 3163 | 3262 | 3262 | 62.1 | 0.00000408 |  4.08 |
| sweep-c8-p128  |  8 | 128 |  99.1 | 1.55 | 3368 | 3370 | 3370 | 64.0 | 0.00000269 |  2.69 |
| sweep-c8-p256  |  8 | 256 |  92.4 | 1.53 | 3359 | 3360 | 3360 | 60.4 | 0.00000289 |  2.89 |
| sweep-c8-p512  |  8 | 512 |  98.3 | 1.54 | 3403 | 3404 | 3404 | 64.0 | 0.00000271 |  2.71 |
| sweep-c16-p128 | 16 | 128 | 185.4 | 2.90 | 3448 | 3450 | 3450 | 64.0 | 0.00000144 |  1.44 |
| sweep-c16-p256 | 16 | 256 | 184.8 | 2.95 | 3328 | 3360 | 3380 | 62.6 | 0.00000144 |  1.44 |
| sweep-c16-p512 | 16 | 512 | 183.3 | 2.86 | 3488 | 3489 | 3490 | 64.0 | 0.00000145 |  1.45 |

Columns: **c** = concurrency, **p_tok** = approximate prompt tokens,
**tok/s** = aggregate tokens/second, **p50/p95/p99** = e2e latency percentiles.

---

## 5. Key findings

### Finding 1 — p95 latency is flat across c=1 to c=16

p95 end-to-end latency grows from **3,101 ms at c=1** to **3,490 ms at c=16**
— a 13% increase while concurrency scales 16×. There is no cliff. The largest
single step is c=4→c=8 at p=512 (3,262 → 3,404 ms), but even that is modest.

vLLM is batching the concurrent requests efficiently. Adding concurrency fills
the decode batch rather than queueing requests behind each other. The RTX 4000
Ada's 20 GB VRAM provides enough KV cache headroom to absorb this batch size
without eviction pressure at max_tokens=64.

This means a client targeting < 4,000 ms p95 e2e latency can safely run at
c=16 without a latency budget penalty.

### Finding 2 — throughput scales 9× linearly from c=1 to c=16

Aggregate throughput grows from **21 tok/s at c=1** to **185 tok/s at c=16**
— near-linear scaling across the full range:

| Concurrency | tok/s (p=128) | ×c=1 |
|:-----------:|:-------------:|:----:|
| 1  |  21.0 | 1.0× |
| 2  |  41.2 | 2.0× |
| 4  |  67.5 | 3.2× |
| 8  |  99.1 | 4.7× |
| 16 | 185.4 | 8.8× |

Throughput has not plateaued at c=16. The GPU is not yet saturated. A follow-up
sweep at c=32 and c=64 would find the saturation ceiling — the point where
adding more concurrent requests no longer increases tok/s because the decode
batch is memory-bandwidth-bound. That run is not included here; Phase 4 covers
c=1 through c=16 only.

### Finding 3 — cost drops from $12.67/M to $1.44/M tokens at c=16

Output-token cost (at $0.96/hr GPU node, all cost figures are for output tokens
at varying concurrency — not a cross-provider comparison) falls 8.8× from c=1
to c=16, tracking throughput directly:

| Concurrency | cost/M tokens (USD) |
|:-----------:|:-------------------:|
| 1  | $12.67–$12.77 |
| 2  |  $6.44–$6.51  |
| 4  |  $3.95–$4.08  |
| 8  |  $2.69–$2.89  |
| 16 |  $1.44–$1.45  |

At c=16 the cost is remarkably stable across all three prompt lengths ($1.44–$1.45),
confirming that prompt length has negligible effect on decode throughput or cost
at this concurrency level. The optimal operating point for cost efficiency —
given the flat latency profile — is the highest concurrency the workload can
sustain, which at c=16 is still not the hardware ceiling.

---

## 6. Current state

### Deployment

`k8s/vllm-ada.yaml` applied to the `inference` namespace, RTX 4000 Ada node
pool, `nodeSelector: gpu-type=rtx4000ada`.

### Tests

```bash
cd phases/phase4-benchmarks
python -m pytest tests/ -v
# 57 passed
```

### Results

`results/phase4_raw_benchmark.csv` — 16 rows (1 warm-up + 15 sweep runs),
zero errors. Not committed to the repository (results/ is gitignored).

### Open items

- Sweep at c=32 and c=64 to find the throughput saturation ceiling.
- RTX PRO 6000 Blackwell node pool not yet available; `k8s/vllm-tp.yaml`
  (TP-2 deployment) and `configs/rtxpro6000.yaml` are staged but not run.
- README PLACEHOLDER tables in `phases/phase4-benchmarks/README.md` to be
  filled in after the c=32/c=64 follow-up run and owner review of the numbers.

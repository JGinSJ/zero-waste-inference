# Phase 4 — Multi-GPU Benchmarking and Cost Model

> **This is the pre-implementation scope document — the plan, not the build.**
> Phase 4 was rescoped to a single-GPU Ada baseline because the RTX PRO 6000
> Blackwell node pool was not available at implementation time, and the
> tensor-parallel target needs two GPUs on one node. For what actually ran, see
> [`docs/phase4-build-log.md`](../phase4-build-log.md) and
> [`phases/phase4-benchmarks/README.md`](../../phases/phase4-benchmarks/README.md).
>
> What changed:
> - **The async SSE harness was not used.** The measured sweep ran on
>   `benchmark/benchmark.py` — synchronous, `requests` +
>   `ThreadPoolExecutor`, non-streaming `/v1/completions` only. `harness/load_gen.py`
>   and `harness/run_benchmark.py` are retained but unexercised.
> - **No TTFT or ITL** in the measured results — those need the SSE path.
> - **Ada pricing is confirmed**, not PLACEHOLDER: `$0.96/hr` for
>   `g2-gpu-rtx4000a1-l`. Blackwell pricing is still unknown.
> - **The multi-GPU comparison is staged, not measured.** `k8s/vllm-tp.yaml` and
>   `configs/rtxpro6000.yaml` remain in place for when the pool exists.

## Goal

Produce a reproducible head-to-head throughput and cost comparison between
RTX 4000 Ada and RTX PRO 6000 Blackwell on Akamai Cloud, and build a
cost model that lets operators estimate the right GPU tier and parallelism
strategy for their workload.

## Inputs and outputs

| | Detail |
|---|---|
| Input | Benchmark config YAML (model, batch sizes, concurrency levels, gpu_hourly_usd) |
| Output | Throughput (tokens/sec), latency (P50/P95/P99 for TTFT, ITL, E2E) |
| Output | cost/token, cost/request, cost/million-tokens — three CSVs |
| Output | `REPORT.md` with latency table, throughput table, cost comparison, TP vs DP guidance |

## Key technologies

- Python 3.11+ benchmarking harness (aiohttp async load generator)
- vLLM OpenAI-compatible endpoint (`/v1/completions`, SSE streaming)
- vLLM `--tensor-parallel-size` for multi-GPU deployment
- **RTX 4000 Ada** on Akamai LKE GPU node pool
- **RTX PRO 6000 Blackwell** on Akamai LKE GPU node pool

## File layout

```
phases/phase4-benchmarks/
├── README.md
├── requirements.txt          # aiohttp, pyyaml, numpy, tabulate, pytest
├── harness/
│   ├── __init__.py
│   ├── run_benchmark.py      # CLI runner — drives all (batch × concurrency) combinations
│   ├── load_gen.py           # Async SSE load generator — records TTFT per request
│   ├── metrics.py            # BenchmarkSummary, LatencyPercentiles, summarise()
│   └── cost_model.py         # compute_cost(), write_csvs() — no prices hardcoded
├── configs/
│   ├── rtx4000ada.yaml       # RTX 4000 Ada config (PLACEHOLDER pricing)
│   └── rtxpro6000.yaml       # RTX PRO 6000 Blackwell config (PLACEHOLDER pricing)
├── k8s/
│   ├── vllm-tp.yaml          # vLLM Deployment with --tensor-parallel-size 2
│   ├── service.yaml          # ClusterIP on 8000
│   ├── benchmark-job-ada.yaml
│   └── benchmark-job-blackwell.yaml
├── report/
│   └── generate_report.py    # Reads results/*.json → REPORT.md (no harness dep)
├── tests/
│   ├── __init__.py
│   ├── test_cost_model.py    # 30 unit tests — cost arithmetic, CSV structure, edge cases
│   └── test_benchmark.py     # 27 unit tests — added with benchmark.py (57 total)
└── results/                  # gitignored — generated outputs land here

(As built, `benchmark/benchmark.py` was added alongside `harness/` and is what
produced the measured results — see the note at the top of this file.)
```

## What the benchmark measures

1. **Throughput** — tokens generated per second at batch sizes 32, 128, 256
   (Blackwell also tests 512).
2. **TTFT** — time-to-first-token via SSE streaming; dominated by prefill latency.
3. **ITL** — inter-token latency: `(e2e - ttft) / (tokens - 1)`; reflects decode speed.
4. **E2E latency** — total wall time per request.
5. **Cost per token** — `gpu_hourly_usd / 3600 / tokens_per_second`.

## Decisions

| Decision | Resolution |
|---|---|
| Cost model methodology | All three metrics: cost/token, cost/request, cost/million-tokens; all derived from a single `gpu_hourly_usd` input |
| GPU pricing source | Ada **confirmed at `$0.96/hr`** (`g2-gpu-rtx4000a1-l`, as of 2026-04) and set in `configs/rtx4000ada.yaml`. Blackwell remains PLACEHOLDER in `configs/rtxpro6000.yaml` — fill in from Akamai published pricing before running that cost model |
| Tensor-parallel strategy | Documented in `phases/phase4-benchmarks/README.md`; decision table defers GPU-specific cells to measured results |
| Concurrency levels | Ada: 1, 4, 8, 16; Blackwell: 1, 4, 8, 16, 32 |

## Open questions

> ~~TODO: Confirm Akamai GPU node hourly pricing for both GPU types~~ —
> **Ada resolved:** `$0.96/hr`, set in `configs/rtx4000ada.yaml`. Blackwell
> pricing is still outstanding in `configs/rtxpro6000.yaml`.

> TODO: Confirm Akamai LKE node pool label names for nodeSelector in
> `k8s/vllm-tp.yaml` and both benchmark Job manifests.

> TODO: Confirm whether RTX PRO 6000 Blackwell nodes have NVLink or PCIe
> between GPUs — this determines the all-reduce overhead for TP-2 runs.
> See the TP vs DP section in `phases/phase4-benchmarks/README.md`.

> TODO: Confirm LKE cluster region and whether both GPU types are available
> in the same region (avoids cross-region latency skewing results).

## Success criteria

- [x] `python -m pytest tests/ -v` passes (57 tests as built — 27 benchmark + 30 cost-model, no GPU required).
- [ ] `run_benchmark.py` completes on both GPU targets without error.
- [ ] CSVs contain non-zero cost figures (requires `gpu_hourly_usd` set).
- [ ] `generate_report.py` produces a valid `REPORT.md` from results.
- [ ] No performance claims appear without a corresponding measured result.
- [ ] All PLACEHOLDER values in configs and K8s manifests replaced before
  production deployment.

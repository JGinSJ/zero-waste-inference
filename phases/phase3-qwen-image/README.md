# Phase 3 — Qwen-Image Inference on Akamai Cloud

End-to-end image-language model serving using Qwen2.5-VL, deployed on
Akamai LKE with RTX 4000 Ada or RTX PRO 6000 Blackwell GPU nodes.

## Live Results

First live inference completed on Akamai LKE (2026-04-14).
Full result: [`results/phase3_baseline_first_result.json`](results/phase3_baseline_first_result.json)

| Field | Value |
|---|---|
| Cluster | akamai-lke-us-ord |
| Node pool | g2-gpu-rtx4000a1-l (RTX 4000 Ada ×1) |
| Model | Qwen/Qwen2.5-VL-7B-Instruct |
| Input | 320×240 solid color JPEG (synthetic test image) |
| Prompt | "Describe what you see." |
| Baseline latency (single request) | **3434.29 ms** |
| Optimizations enabled | none (`USE_OPTIMIZED=false`) |

This is the baseline measurement.

### Optimized results

Full results: [`results/phase3_optimized_bench.json`](results/phase3_optimized_bench.json)

Active optimizations: `OPTIMIZED=1` (bfloat16 + sdpa + torch.compile, full bundle)

| Metric | Value |
|---|---|
| Optimizations | bfloat16 · sdpa · torch.compile (reduce-overhead) |
| dtype (from /health) | bfloat16 |
| Runs measured | 9 (run 1 discarded — torch.compile warm-up) |
| Server latency mean | **2,069 ms** |
| Server latency std | 2.94 ms |
| Server latency min / max | 2,067 ms / 2,075 ms |
| Warm-up run latency | 2,874 ms (run 1, discarded) |
| **vs. baseline** | **−1,365 ms (−39.7%)** |

The three optimizations were enabled as a bundle (`OPTIMIZED=1`). Per-optimization
isolation would require multiple pod restarts on a live cluster and was not measured.
The −39.7% figure represents the combined effect of bfloat16, sdpa, and torch.compile.

---

## Two serving paths

```
POST /v1/generate
       │
       ├── USE_OPTIMIZED=0 (default)
       │       serve/model.py → load_model()
       │       float16 · standard attention · no compile
       │       Deterministic baseline for benchmarking.
       │
       └── USE_OPTIMIZED=1  [EXPERIMENTAL]
               serve/model_optimized.py → load_model_optimized()
               Optional flags: flash_attention_2 · bfloat16 · torch.compile
               Measure results before drawing conclusions — see below.
```

Both paths share the same `run_inference()` function and produce
comparable outputs (minor numerical differences possible between dtypes).

## What "EXPERIMENTAL" means

Each optimization in `serve/model_optimized.py` is labelled EXPERIMENTAL
because its effect depends on GPU architecture, sequence length, and batch
composition.  No throughput or latency improvement is claimed.

| Optimization | Mechanism | Notes |
|---|---|---|
| `sdpa` | `attn_implementation="sdpa"` — PyTorch's built-in `scaled_dot_product_attention`, dispatched to fused CUDA kernels on Ada/Blackwell at runtime | No extra package required; enabled via `USE_FLASH_ATTN=1` |
| `bfloat16` | Larger float dynamic range vs float16 | May suit Blackwell's tensor cores; measured −39.7% on Ada as part of the full bundle |
| `torch.compile` | Traces compute graph, emits optimized kernels | Slow first call (~2,874 ms warm-up); stable at ~2,069 ms from run 2 onward |

**How to measure:** run `benchmark/load_gen.py` with `USE_OPTIMIZED=0`,
save results as `phase3_baseline.json`, repeat with `USE_OPTIMIZED=1` as
`phase3_optimized.json`, compare with `benchmark/report.py --baseline ...
--optimized ...`.  Let the numbers speak.

---

## Model

Default: `Qwen/Qwen2.5-VL-7B-Instruct`

| Variant | Approx VRAM (FP16) |
|---|---|
| Qwen/Qwen2.5-VL-3B-Instruct | ~ 8 GB |
| **Qwen/Qwen2.5-VL-7B-Instruct** ← default | ~16 GB |
| Qwen/Qwen2.5-VL-72B-Instruct | ~144 GB |

Override at runtime: `MODEL_NAME=Qwen/Qwen2.5-VL-3B-Instruct uvicorn ...`

---

## File layout

```
phase3-qwen-image/
├── serve/
│   ├── __init__.py           # Exports only batching layer (no ML deps at import time)
│   ├── app.py                # FastAPI server, /v1/generate + /health
│   ├── model.py              # Baseline loader + run_inference()
│   ├── model_optimized.py    # EXPERIMENTAL optimized loader
│   ├── batching.py           # DynamicBatcher, Future, InferenceRequest
│   └── image_utils.py        # decode_image() — stdlib + Pillow only, no torch/transformers
├── k8s/
│   ├── deployment.yaml       # LKE Deployment (GPU, ConfigMap)
│   ├── service.yaml          # ClusterIP on port 8080
│   └── gpu-node-pool.yaml    # Akamai LKE node pool spec (PLACEHOLDER)
├── benchmark/
│   ├── requirements.txt
│   ├── load_gen.py           # Synthetic-image load generator
│   └── report.py             # Throughput + latency analysis
└── tests/
    ├── __init__.py
    └── test_model.py         # Infrastructure tests (no GPU); GPU tests skipped without CUDA
```

---

## Setup

```bash
cd phases/phase3-qwen-image
pip install -r requirements.txt
```

Requires Python 3.11+, PyTorch 2.4+, and `transformers >= 4.45, < 5.0`.

**Flash Attention 2 (EXPERIMENTAL, optional):**
```bash
pip install flash-attn --no-build-isolation
```
Requires a CUDA toolchain. Not needed for the baseline path.

---

## Local development

`torch >= 2.4` wheels are not published for Intel Mac (x86_64 macOS).
PyTorch dropped Intel Mac binary releases after 2.2.x.

What this means in practice:

| Task | Intel Mac | Akamai LKE (Linux x86_64 + CUDA) |
|---|---|---|
| `pip install -r requirements.txt` | Fails on `torch>=2.4` | Succeeds |
| Import verification (`from transformers import ...`) | OK — install `transformers<5.0` alone | OK |
| Infrastructure tests (batcher, Future, image decode) | OK — no torch required | OK |
| Full server startup (`uvicorn serve.app:app`) | Blocked — torch unavailable | OK |
| Model loading and inference | Not possible | OK |

**Working locally on Intel Mac:**

Install only the non-torch dependencies to run the infrastructure tests:

```bash
pip install "transformers>=4.45,<5.0" Pillow fastapi uvicorn pydantic pytest
python -m pytest tests/ -v
# 18 tests pass; 4 GPU tests correctly skipped
```

Full server startup and model loading require the Akamai LKE GPU deployment
target. See [Deploy to LKE](#deploy-to-lke) below.

---

## Run the server locally

```bash
# Baseline
uvicorn serve.app:app --host 0.0.0.0 --port 8080

# EXPERIMENTAL optimized path
USE_OPTIMIZED=1 USE_FLASH_ATTN=1 uvicorn serve.app:app --port 8080

# Without a GPU (CPU inference — very slow, for development only)
MODEL_NAME=Qwen/Qwen2.5-VL-3B-Instruct uvicorn serve.app:app --port 8080
```

## Request / response

```
POST /v1/generate
{
  "image":          "<base64-encoded JPEG or PNG>",
  "prompt":         "Describe this image.",
  "max_new_tokens": 256
}

→ 200 OK
{
  "response":    "A solid red square on a white background.",
  "latency_ms":  842.5,
  "gpu":         "NVIDIA RTX 4000 Ada Generation",
  "optimized":   false
}
```

---

## Deploy to LKE

```bash
# Edit first:
#   k8s/deployment.yaml — set image, MODEL_NAME, nodeSelector
#   k8s/gpu-node-pool.yaml — provision node pool via Linode API/CLI

kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

kubectl rollout status deployment/qwen-image -n inference
kubectl logs -n inference -l app=qwen-image -f
```

---

## Run the tests

```bash
python -m pytest tests/ -v
```

Tests that run without a GPU (batcher, Future, image decode, schema
validation): **always run**.

Tests that require CUDA: **skipped** with `reason="requires CUDA device"`.
The skip message names the target hardware.  A separately skipped stub
marks the EXPERIMENTAL optimized-path inference test.

---

## Run the benchmark

```bash
cd phases/phase3-qwen-image
pip install -r benchmark/requirements.txt

# Baseline run
python benchmark/load_gen.py \
    --url http://<host>:8080/v1/generate \
    --runs 20 \
    --batch-sizes 1,4,16 \
    --output results/phase3_baseline.json \
    --tag baseline

# EXPERIMENTAL optimized run (restart server with USE_OPTIMIZED=1 first)
python benchmark/load_gen.py \
    --url http://<host>:8080/v1/generate \
    --runs 20 \
    --batch-sizes 1,4,16 \
    --output results/phase3_optimized.json \
    --tag optimized

# Compare
python benchmark/report.py \
    --baseline results/phase3_baseline.json \
    --optimized results/phase3_optimized.json
```

---

## Future work

- **True multi-image batching:** The current batch worker processes requests
  sequentially within a batch. Uniform batched Qwen2.5-VL inference requires
  padding variable-length vision tokens to a common sequence length — a TODO.

- **Phase 2 integration:** Image+prompt responses could be cached in Valkey
  (Phase 2 pipeline) using the same exact-match key scheme —
  `SHA-256(model + "\0" + normalised messages)` over the whole message array,
  not a prompt-prefix hash — eliminating redundant GPU calls for repeated
  queries.  No code changes to Phase 2 are needed; the Phase 3 server would sit
  behind the Fermyon handler as an additional `vllm_url`-equivalent backend.

- **vLLM multimodal mode:** An alternative to raw `transformers` is
  `vllm serve` with a multimodal model, which would unify Phases 2 and 3 under
  a single vLLM deployment.  Note that the Phase 2 deployment gets prefix reuse
  from LMCache with `--no-enable-prefix-caching`; a unified deployment would
  need to pick one mechanism, since the two do not stack.

---

## Success criteria

- [ ] `python -m pytest tests/ -v` passes (GPU tests correctly skipped).
- [ ] Server starts with `uvicorn serve.app:app` and `/health` returns 200.
- [ ] Model loads on target GPU without OOM error.
- [ ] Single-request smoke test returns non-empty `response` string.
- [ ] `benchmark/load_gen.py` completes at batch sizes 1, 4, 16.
- [ ] `benchmark/report.py` prints a valid latency table from results.
- [ ] No performance claims appear without a corresponding measured result.

# Zero-Waste Inference on Akamai Cloud

A production-quality demonstration of inference cost reduction through
computation reuse, deployed on Akamai Cloud (LKE) with NVIDIA RTX 4000 Ada
and RTX PRO 6000 Blackwell GPU targets.

## What this project shows

LLM inference is expensive because compute is repeatedly thrown away.
This project demonstrates four techniques for reusing computation at
different points in the inference stack:

| Phase | Technique | Key Technologies | Status |
|-------|-----------|-----------------|--------|
| 1 | KV cache from scratch | PyTorch transformer demo | Complete |
| 2 | Prefix cache at the edge | Fermyon Wasm, Valkey, vLLM | Complete — live: us-ord |
| 3 | Image model inference | Qwen-Image on Akamai Cloud | Complete — live: us-ord |
| 4 | Single-GPU benchmarking & cost model | RTX 4000 Ada (Blackwell pending) | Complete |

## Repository layout

```
akamai-inference-optimization/
├── infrastructure/
│   ├── README.md               # Provisioning guide and post-cluster checklist
│   └── terraform/              # LKE cluster, CPU pool, GPU node pools
├── phases/
│   ├── phase1-kv-cache/        # PyTorch KV cache demo
│   ├── phase2-prefix-cache/    # Fermyon + Valkey + vLLM  [live: us-ord]
│   ├── phase3-qwen-image/      # Qwen-Image inference      [live: us-ord]
│   └── phase4-benchmarks/      # Single-GPU cost model (Ada measured; Blackwell pending)
├── docs/
│   ├── architecture.md         # System diagram and phase map
│   ├── hardware.md             # GPU target specifications
│   ├── cluster-startup.md      # Operational runbook: scale up/down, port-forwards
│   └── phases/                 # Per-phase scope documents
├── CLAUDE.md                   # AI session working rules
├── LICENSE                     # Apache 2.0
└── pyproject.toml              # Root workspace marker
```

## Prerequisites

- Python 3.11 or later
- pip
- NVIDIA GPU (RTX 4000 Ada or RTX PRO 6000 Blackwell for full benchmarks)
- Access to an Akamai Cloud / LKE cluster (Phase 2+)
- Fermyon Cloud account or self-hosted Spin runtime (Phase 2)

## Quick start

Each phase is self-contained. Navigate to the relevant `phases/` directory
and follow the README there.

```bash
# Phase 1 — KV cache demo (CPU, no GPU required)
cd phases/phase1-kv-cache
pip install -r requirements.txt
python demo.py

# Phase 2 — prefix cache tests (no live cluster required)
cd phases/phase2-prefix-cache
pip install -r benchmark/requirements.txt
python -m pytest tests/ -v

# Phase 3 — infrastructure tests (no GPU required)
cd phases/phase3-qwen-image
pip install "transformers>=4.45,<5.0" Pillow fastapi uvicorn pydantic pytest
python -m pytest tests/ -v

# Phase 4 — cost model tests (no GPU or network required)
cd phases/phase4-benchmarks
pip install -r requirements.txt
python -m pytest tests/ -v
```

## Phases at a glance

### Phase 1 — KV Cache from Scratch

A minimal PyTorch transformer that makes the key-value cache visible and
measurable. No HuggingFace dependencies. Goal: show exactly what is being
reused and what is being recomputed on each forward pass.

See [docs/phases/phase1-kv-cache.md](docs/phases/phase1-kv-cache.md).

### Phase 2 — Fermyon + Valkey + vLLM Prefix Caching

A Fermyon Wasm Function sits at the front door. It hashes the request's model
and messages, checks a Valkey cache, and short-circuits exact repeats before
they reach the GPU. Misses fall through to a vLLM backend where LMCache
handles prefix reuse, offloading KV blocks to the same Valkey instance.

See [docs/phases/phase2-fermyon-valkey.md](docs/phases/phase2-fermyon-valkey.md).

### Phase 3 — Qwen-Image Inference on Akamai Cloud

End-to-end image-language model serving using Qwen2.5-VL, deployed to
Akamai LKE with GPU node pools. Covers model loading, dynamic batching,
baseline and optimised serving paths, and request routing.

See [docs/phases/phase3-qwen-image.md](docs/phases/phase3-qwen-image.md).

### Phase 4 — Single-GPU Benchmarking and Cost Model

Throughput and cost measurement for RTX 4000 Ada on Akamai Cloud. Uses a
synchronous concurrency sweep (1→4→8→16), with a cost model that converts
GPU-hour pricing to cost-per-token and cost-per-million-tokens. RTX PRO 6000
Blackwell comparison is deferred pending node pool activation.

**Ada results summary** (Mistral-7B, prompt≈256 tok, max_tokens=64,
gpu_hourly_usd=$0.96):

| Concurrency | tok/s | cost/M tokens (USD) |
|:-----------:|------:|--------------------:|
| 1           |  20.9 | $12.77              |
| 4           |  67.5 |  $3.95              |
| 8           |  92.4 |  $2.89              |
| 16          | 184.8 |  $1.44              |

See [docs/phases/phase4-benchmarks.md](docs/phases/phase4-benchmarks.md).

## License

Apache 2.0 — see [LICENSE](LICENSE).

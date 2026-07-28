# Phase 2 — Fermyon + Valkey + vLLM Prefix Caching

A prefix-aware inference pipeline where a Fermyon Wasm Function at the
Akamai edge intercepts requests, checks a Valkey cache, and only forwards
cache misses to a vLLM backend.

## Live Results

First live prefix-cache run completed on Akamai LKE (2026-04-14).
Full result: [`results/phase2_prefix_cache_baseline.json`](results/phase2_prefix_cache_baseline.json)

| Field | Value |
|---|---|
| Cluster | akamai-lke-us-ord |
| Node pool | g2-gpu-rtx4000a1-l (RTX 4000 Ada ×1) |
| Model | mistralai/Mistral-7B-Instruct-v0.2 |
| Prompt | "What is prefix caching in LLMs? Answer in 2 sentences." |
| Request 1 (cold) wall clock | 3.003 s — 23 prompt tokens, 63 completion tokens |
| Request 2 (warm) wall clock | 3.385 s — 23 prompt tokens, 71 completion tokens |
| vLLM token cache hit rate | **46.4%** (32 of 69 tokens served from GPU cache) |
| GPU blocks allocated | 1604 |
| Valkey external cache | Deployed; not yet wired as KV connector |

**On the wall-clock similarity between cold and warm requests:** the prompt is
only 23 tokens, so the prefix cache saves a small fraction of prefill work.
Larger speedups appear with longer shared prefixes (e.g. multi-shot system
prompts of several hundred tokens). The 46% token hit rate in the vLLM metrics
confirms the cache is active and serving repeated KV states across requests.

### LMCache + Valkey connector — verified 2026-04-15

Valkey is now wired as the external KV connector via LMCache 0.4.3 on vLLM 0.18.1.
Full result: [`results/phase2_valkey_verified.json`](results/phase2_valkey_verified.json)

> **Note:** this benchmark targets vLLM directly (pre-Fermyon-deployment).
> The Fermyon front door is not yet wired to this LMCache-backed instance.

| Field | Value |
|---|---|
| vLLM version | 0.18.1 |
| LMCache version | 0.4.3 |
| Connector | LMCacheConnectorV1 |
| Backend | `resp://valkey-svc.inference.svc.cluster.local:6379` |
| Store: 256/256 tokens | 3.70 ms — 8.46 GB/s |
| Retrieve: 256/256 tokens | 1.56 ms — 20.07 GB/s |
| External prefix cache hit rate | **25.9%** (confirmed in vLLM metrics) |

---

## Why this reduces inference cost

Phase 1 showed that KV caching saves compute *within* a single model run.
Phase 2 saves compute *across* runs by caching entire responses.

### Two complementary caching layers

```
Request
  │
  ▼
Fermyon Wasm Function          ← Layer 1: response cache (Valkey)
  │  hash(model + messages)
  │  GET from Valkey
  ├── HIT  ──────────────────────► return cached response  (0 GPU work)
  │
  └── MISS ─────────────────────► vLLM backend
                                     │  LMCacheConnectorV1
                                     │                          ← Layer 2: KV cache (GPU → Valkey overflow)
                                     │  requests sharing a prefix
                                     │  reuse cached KV states
                                     ▼
                                  GPU generates response
                                     │
                                     └── store in Valkey ──► return to client
```

**Layer 1 (Fermyon → Valkey):** Full response cache, **exact match**.  If the
identical `(model, messages)` pair was seen before, return the stored answer
immediately — no GPU compute at all.  This layer does *not* match on shared
prefixes; see the key derivation below.

**Layer 2 (vLLM → LMCache → Valkey):** KV-state cache.  For cache misses that
share a prompt prefix (e.g. a system prompt), the already-computed attention
key/value states for the shared prefix are reused, reducing prefill cost.  This
is the same mechanism demonstrated from scratch in Phase 1.  Note that vLLM's
*built-in* prefix cache is switched off (`--no-enable-prefix-caching`) because
LMCache manages this externally and the two do not stack.

### Cache key derivation

The Valkey key written by the Fermyon proxy is:

```
"fermyon:v1:" + hex( SHA-256( model + "\0" + messages_json ) )
```

75 characters total (`fermyon:v1:` + a 64-char hex digest).  `messages` is
normalised to `[{role, content}]` first, so field order and extra fields
(`name`, `tool_call_id`, …) do not change the key.

Only `model` and `messages` contribute.  Sampling parameters — `temperature`,
`top_p`, `max_tokens`, `stream` — are deliberately excluded, so requests with
identical prompt content share an entry even when their sampling knobs differ.

**This is an exact-match response cache, not a prefix cache.**  The digest
covers the whole message array: two requests that share a long opening but
diverge by a single token at the end produce different keys and do not share an
entry.  Prefix-level reuse happens one layer down, in LMCache.  See
[Future work](#future-work) for the semantic caching path.

Entries are written with a TTL (`SPIN_VARIABLE_CACHE_TTL`, default 3600 s),
applied via `EXPIRE` on write.  The TTL is **not** reset on hits — it counts
down from first write.  Both the `SET` and the `EXPIRE` are best-effort: a
failure never fails the client request, it just means the next identical
request is another miss.

---

## Components

| Component | Technology | Where it runs |
|---|---|---|
| Front door | Fermyon Wasm (Rust, Spin 3.6.3) | Akamai LKE (CPU node, `workload-type=cpu`) |
| Response cache | Valkey 8.0 standalone | Akamai LKE (CPU node) |
| Inference backend | vLLM 0.18.1 + LMCache 0.4.3 (`LMCacheConnectorV1`) | Akamai LKE (GPU node, `gpu-type=rtx4000ada`) |

---

## File layout

```
phase2-prefix-cache/
├── fermyon/
│   ├── Cargo.toml          # workspace root — no [package]
│   ├── spin.toml           # Spin app manifest — one trigger per component
│   ├── Dockerfile          # debian:bookworm-slim + pinned Spin v3.6.3
│   ├── proxy/
│   │   └── src/lib.rs      # POST /v1/chat/completions — hash → Valkey → vLLM
│   ├── health/
│   │   └── src/lib.rs      # GET /health — no outbound hosts
│   ├── src/lib.rs          # legacy single-crate handler (superseded, unused)
│   └── k8s/
│       └── fermyon-deployment.yaml   # Namespace + ConfigMap + Deployment + Service
├── valkey/
│   ├── valkey.yaml         # LKE Deployment + Service + ConfigMap
│   └── config/
│       └── valkey.conf     # Standalone, allkeys-lru, 2 GB cap
├── vllm/
│   ├── Dockerfile              # vllm/vllm-openai:v0.18.1 + lmcache==0.4.3
│   ├── lmcache-configmap.yaml  # LMCache config — Valkey RESP backend
│   ├── vllm.yaml               # LKE Deployment + Service (GPU node)
│   ├── pvc-model-cache.yaml    # PVC for HuggingFace model weights
│   └── serve_config.yaml       # vLLM flags reference (--no-enable-prefix-caching + LMCache)
├── benchmark/
│   ├── requirements.txt
│   ├── bench_cache.py      # Three-pass cache-value benchmark (MISS / HIT / direct vLLM)
│   ├── load_gen.py         # Send N requests with configurable prefix-share rate (broken against live endpoint — see note below)
│   └── report.py           # Hit rate + latency report from load_gen output (broken against live endpoint — see note below)
└── tests/
    ├── test_prefix_hash.py # Legacy hash contract tests + semantic-cache stub (14 pass, 1 skipped)
    └── test_chat_cache.py  # Live cache-key contract (21 unit + 4 integration stubs)
```

---

## Setup

### Prerequisites

| Tool | Purpose |
|---|---|
| Rust + `wasm32-wasip1` target | Build the Fermyon Wasm binary |
| `spin` CLI | Run and deploy the Fermyon app |
| `kubectl` | Apply Valkey and vLLM manifests to LKE |
| Python 3.11+ | Benchmark and tests |

Install the Rust Wasm target:
```bash
rustup target add wasm32-wasip1
```

Install Spin CLI: follow [developer.fermyon.com](https://developer.fermyon.com/spin/install).

### Build the Wasm handler

```bash
cd phases/phase2-prefix-cache/fermyon
cargo build --workspace --target wasm32-wasip1 --release
```

Two binaries are written to the workspace-level `target/`:

```
target/wasm32-wasip1/release/proxy.wasm
target/wasm32-wasip1/release/health.wasm
```

`--workspace` matters — building without it misses one of the two components,
and the paths above must match the `source =` values in `spin.toml`.

### Run locally with Spin

```bash
cd phases/phase2-prefix-cache/fermyon
spin up --listen 0.0.0.0:8082 \
  --variable valkey_address=redis://localhost:6379 \
  --variable vllm_url=http://localhost:8000 \
  --variable cache_ttl=3600
```

The proxy listens on `http://localhost:8082/v1/chat/completions`, with the
health component on `/health`.  The listen address is a **runtime flag** in
Spin 3.x — it is not a `spin.toml` field.

### Deploy to LKE

```bash
# Create the namespace and deploy Valkey
kubectl apply -f phases/phase2-prefix-cache/valkey/valkey.yaml

# Build and push the vLLM + LMCache image
docker build -t ghcr.io/jginsj/vllm-lmcache:v0.18.1 \
    -f phases/phase2-prefix-cache/vllm/Dockerfile \
    phases/phase2-prefix-cache/vllm/
docker push ghcr.io/jginsj/vllm-lmcache:v0.18.1

# Apply LMCache config and PVC before the deployment
kubectl apply -f phases/phase2-prefix-cache/vllm/lmcache-configmap.yaml
kubectl apply -f phases/phase2-prefix-cache/vllm/pvc-model-cache.yaml

# Deploy vLLM with LMCache connector
kubectl apply -f phases/phase2-prefix-cache/vllm/vllm.yaml

# Build and push the Fermyon image, then deploy it to the CPU node.
# (The front door runs as a container on LKE — not on Fermyon Cloud.)
cd phases/phase2-prefix-cache/fermyon
cargo build --workspace --target wasm32-wasip1 --release
docker build -t ghcr.io/jginsj/fermyon-prefix-cache:latest -f Dockerfile .
docker push ghcr.io/jginsj/fermyon-prefix-cache:latest

# The CPU node must carry the workload-type label before the pod can schedule
kubectl label node <cpu-node-name> workload-type=cpu --overwrite

kubectl apply -f k8s/fermyon-deployment.yaml
```

> **GHCR visibility:** new packages default to **private**, and the nodes have
> no registry credentials — the pull will fail until the package is set to
> public in the GitHub UI (Settings → Packages → Change visibility).

Runtime configuration comes from the `fermyon-config` ConfigMap, which Spin
reads as `SPIN_VARIABLE_*` environment variables: `VALKEY_ADDRESS`,
`VLLM_URL`, `CACHE_TTL`.

**Before deploying vLLM:** `MODEL_NAME` and `nodeSelector` are pre-configured
in `vllm/vllm.yaml` for the us-ord cluster. Verify before applying to a
different cluster:
- `MODEL_NAME`: `mistralai/Mistral-7B-Instruct-v0.2`
- nodeSelector: `gpu-type: rtx4000ada`

---

## Request / response format

The proxy is an OpenAI-compatible **transparent passthrough**.  It speaks the
same request and response shape as vLLM; the only additions are two response
headers.

**Request** — standard chat completions:
```json
POST /v1/chat/completions
{
  "model": "mistralai/Mistral-7B-Instruct-v0.2",
  "messages": [{"role": "user", "content": "What is 2+2?"}],
  "max_tokens": 64
}
```

Unknown fields are preserved and forwarded to vLLM unchanged.  `stream: true`
is the exception: it is stripped before forwarding (the Wasm runtime cannot do
streaming responses) and the event is logged to stderr.

**Response** — the raw vLLM JSON body, unmodified, plus:

```
Content-Type: application/json
X-Cache: HIT | MISS
```

`X-Cache: HIT` means the body came from Valkey without touching the GPU.
There is no `cache_hit` field in the body — cache state travels in the header,
which is why `benchmark/load_gen.py` and `report.py` no longer work against
the live endpoint (see [Known issues](#known-issues)).

---

## Cache benchmark results

Measured 2026-04-16 on Akamai LKE us-ord, RTX 4000 Ada node.
Full results: [`results/phase2_cache_benchmark.json`](results/phase2_cache_benchmark.json)

Three-pass measurement: cold cache (Pass 1), warm cache (Pass 2), direct vLLM
baseline (Pass 3). 10 sequential requests per pass, `max_tokens=64`,
shared prefix ≈ 500 tokens. 0 errors across 30 requests.

| Pass | p50 (ms) | p95 (ms) |
|---|---|---|
| Pass 1 — MISS (Fermyon → vLLM) | 3,058 | 5,897 * |
| Pass 2 — HIT  (Fermyon → Valkey) | 218 | 221 |
| Pass 3 — Direct vLLM (no cache) | 3,020 | 3,079 |

\* Pass 1 p95 is inflated by an 8,173 ms first-request spike (request 1 of 10),
consistent with vLLM cold-start on the first inference after pod readiness.
Requests 2–10 all fell in the 2,820–3,116 ms range. The p50 (3,058 ms) is
unaffected and is the correct MISS latency for the break-even calculation.

### Cache value

```
Miss overhead  = Pass1 p50 − Pass3 p50  =  3,058 − 3,020  =   +38 ms
Hit saving     = Pass3 p50 − Pass2 p50  =  3,020 −   218  = 2,802 ms

Break-even hit rate = miss_overhead / (miss_overhead + hit_saving)
                    =      38       / (     38       +   2,802   )
                    =   1.3%
```

The miss overhead (+38 ms) is the Valkey round-trip cost on every cache miss —
Fermyon misses cost slightly more than direct vLLM. The break-even hit rate of
1.3% means the cache layer produces net-positive latency impact at any realistic
hit rate above that floor.

**What the break-even does not tell you.** It says the Fermyon layer pays for
itself at almost any realistic repeat rate. It does not say what that repeat
rate is — the Fermyon hit rate was never measured under production-shaped
traffic, only in Pass 2 above, which was constructed to hit.

The 25.9% figure recorded in `results/phase2_valkey_verified.json` is **not**
this layer's hit rate: it belongs to LMCache's external KV cache, measured
against vLLM directly on 2026-04-15, before the Fermyon front door was wired to
that instance. Two different caches, two different runs — do not compare it
against the 1.3% break-even above.

### Running the benchmark

```bash
cd phases/phase2-prefix-cache
pip install -r benchmark/requirements.txt

python benchmark/bench_cache.py \
    --fermyon-url http://localhost:8082 \
    --vllm-url    http://localhost:8000
```

---

## Running the tests

Neither test file requires running infrastructure.

```bash
cd phases/phase2-prefix-cache
python -m pytest tests/ -v
# 35 passed, 1 skipped, 4 deselected
```

**`test_chat_cache.py`** — the contract for the key the live proxy actually
writes (21 unit tests, plus 4 integration stubs deselected by default):
- Key format: `fermyon:v1:` + 64-char lowercase hex, 75 characters total
- Determinism: same `(model, messages)` → same key, always
- Sampling params (`temperature`, `top_p`, `max_tokens`, `stream`) excluded
- Field-order independence after message normalisation
- Null-byte separator prevents `(model, messages)` boundary collisions
- Pinned SHA-256 values as a cross-language compatibility check

**`test_prefix_hash.py`** — contract tests for the earlier `prompt[:128]`
scheme (14 pass, 1 skipped). That scheme is **superseded** by the key above and
is retained only for the semantic-cache stub,
`test_semantic_cache_equivalent_prompts_share_key`, which is skipped and marks
the semantic caching work as not yet implemented.

The 4 integration tests need a live endpoint:

```bash
kubectl port-forward svc/fermyon-svc 8082:8082 -n inference &
FERMYON_URL=http://localhost:8082 python -m pytest tests/test_chat_cache.py -v -m integration
```

---

## Future work

### Semantic caching (TODO)

The current implementation is **exact-match**: two requests share a cache entry
only if their `(model, messages)` pair is byte-for-byte identical after
normalisation.

A future implementation would:
1. Embed the prompt prefix with a sentence-transformer model.
2. Query a vector store (e.g. pgvector, Qdrant) for the nearest cached
   embedding within a similarity threshold.
3. Return the nearest neighbour's cached response if similarity exceeds the
   threshold; otherwise proceed as a cache miss.

This would catch semantically equivalent prompts that are worded differently
(rephrased system prompts, synonym substitutions, language variants).

Tracking: `tests/test_prefix_hash.py::test_semantic_cache_equivalent_prompts_share_key`
(currently skipped with a full TODO explanation).

### Other deferred items

- Valkey cluster mode for horizontal scale and HA
- TLS between Fermyon and Valkey
- Streaming responses (blocked on Wasm runtime support — `stream: true` is
  currently stripped)
- Sliding-window TTL — entries currently expire a fixed 3600 s after first
  write and are not refreshed on hits
- CI/CD pipeline for Wasm build and LKE deploy
- `kubectl` wait / health-check scripts for the deploy sequence

---

## Known issues

`benchmark/load_gen.py` and `benchmark/report.py` are broken against the live
Fermyon endpoint. They expect a `cache_hit` field in the JSON response body, but
the live Fermyon handler signals cache state via the `X-Cache: HIT|MISS` response
header. Use `bench_cache.py` for all live-cluster measurements.

---

## Success criteria

- [x] `cargo build --workspace --target wasm32-wasip1 --release` succeeds.
- [x] Valkey pod is healthy in LKE (`kubectl get pods -n inference`).
- [x] vLLM pod starts with LMCacheConnectorV1 and Valkey backend confirmed in logs.
- [x] Valkey store/retrieve verified: 256/256 tokens, 3.70 ms store, 1.56 ms retrieve.
- [x] LMCache external KV hit rate: 25.9% confirmed in vLLM metrics (this is
      Layer 2, measured direct-to-vLLM — not the Fermyon response cache).
- [x] `python -m pytest tests/ -v` passes — 35 passed, 1 skipped (semantic stub).
- [x] Cached requests show lower median latency than uncached: 218 ms vs 3,020 ms p50.
- [ ] Fermyon-layer hit rate measured under production-shaped traffic.
- [ ] ~~Load generator produces requests with a 50 % shared-prefix rate.~~ —
      blocked: `load_gen.py` / `report.py` read a `cache_hit` body field the
      live handler does not emit. See [Known issues](#known-issues).
- [ ] Streaming responses supported end to end.

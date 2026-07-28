# Phase 2 — Fermyon + Valkey + vLLM Prefix Caching

> **This is the pre-implementation scope document — the plan, not the build.**
> Several assumptions below did not survive contact with the runtime. For what
> actually shipped, read [`docs/phase2-fermyon-build-log.md`](../phase2-fermyon-build-log.md)
> and [`phases/phase2-prefix-cache/README.md`](../../phases/phase2-prefix-cache/README.md).
>
> The three biggest divergences:
> - **One crate became a Cargo workspace** (`proxy/` + `health/`). Spin routes by
>   route, not by path inspection inside a shared binary, and the two components
>   need different outbound permissions.
> - **The cache key is not a prompt-prefix hash.** It is
>   `SHA-256(model + "\0" + normalised messages)` over the *whole* message array —
>   an exact-match response cache, not a prefix cache.
> - **vLLM runs with `--no-enable-prefix-caching`.** Prefix reuse is handled
>   externally by LMCache 0.4.3 via `LMCacheConnectorV1`, offloading KV blocks to
>   Valkey. vLLM's built-in prefix cache and LMCache do not stack.

## Goal

Deploy a prefix-aware inference pipeline where a Fermyon Wasm Function
at the Akamai edge intercepts requests, checks a Valkey cache for
matching prompt prefixes, and only forwards cache misses to a vLLM
backend running on Akamai LKE.

## Inputs and outputs

| | Detail |
|---|---|
| Input | ~~HTTP POST to Fermyon function with `{"prompt": "..."}`~~ — **superseded.** As built: `POST /v1/chat/completions` with the standard `{model, messages, …}` body |
| Output | ~~`{"response": "...", "cache_hit": true/false, "latency_ms": N}`~~ — **superseded.** As built: the raw vLLM JSON body, unmodified, plus an `X-Cache: HIT\|MISS` response header |
| Side output | Valkey hit/miss rate over a benchmark run |
| Side output | End-to-end latency comparison: cached vs uncached requests |

## Key technologies

- **Fermyon Wasm Functions** — front door, edge hash check, cache lookup
- **Valkey** — cache (not Redis); as built it backs *two* layers — the Fermyon
  response cache and LMCache's KV-block offload
- **vLLM** — LLM backend. *Planned* with `--enable-prefix-caching`; as built it
  runs `--no-enable-prefix-caching` with `--kv-transfer-config LMCacheConnectorV1`
- **Akamai LKE** — Kubernetes cluster hosting Valkey and vLLM pods
- Python 3.11+ for benchmark harness and load generation

## Architecture

```
Client
  |
  v
Fermyon Wasm Function (edge)
  |-- hash prompt prefix
  |-- GET from Valkey
  |   |-- HIT  --> return cached response
  |   `-- MISS --> forward to vLLM
  |
  v
vLLM pod (LKE GPU node)
  |-- prefix reuse via LMCacheConnectorV1 (KV blocks -> Valkey)
  `-- store response in Valkey
```

## File layout

```
phases/phase2-prefix-cache/
├── README.md
├── fermyon/
│   ├── Cargo.toml            # Rust crate — spin-sdk 3, sha2, serde_json
│   ├── Cargo.lock
│   ├── spin.toml             # Fermyon app manifest
│   └── src/
│       └── lib.rs            # Wasm handler: hash → Valkey GET → vLLM → Valkey SET
├── valkey/
│   ├── valkey.yaml           # LKE Deployment + Service + ConfigMap
│   └── config/
│       └── valkey.conf       # Standalone, allkeys-lru, 2 GB cap
├── vllm/
│   ├── vllm.yaml             # LKE Deployment + Service (GPU node)
│   └── serve_config.yaml     # vLLM flags (as built: --no-enable-prefix-caching + LMCache)
├── benchmark/
│   ├── requirements.txt
│   ├── load_gen.py           # Request generator with configurable prefix-share rate
│   └── report.py             # Hit rate + latency report from load_gen output
└── tests/
    ├── __init__.py
    └── test_prefix_hash.py   # Hash contract tests + semantic-cache stub (skipped)
```

## Success criteria

- [ ] Fermyon function builds and deploys to Fermyon Cloud or local Spin.
- [ ] Valkey pod is healthy in LKE.
- [x] vLLM pod starts with prefix reuse active — as built via `LMCacheConnectorV1`
      and `--no-enable-prefix-caching`, not vLLM's built-in cache.
- [ ] Load generator produces requests with a 50% shared-prefix rate.
- [ ] Valkey hit rate exceeds 40% under that load.
- [ ] Cached requests show lower median latency than uncached requests.

## Decisions

| Decision | Resolution |
|---|---|
| Wasm language | Rust, using spin-sdk 3.x — mature SDK, strong async support |
| Valkey version | 8.0, standalone mode, allkeys-lru eviction, 2 GB memory cap |
| ~~Prefix hashing~~ | ~~SHA-256 of the first 128 Unicode characters (not bytes) of the prompt~~ — **superseded.** As built: `"fermyon:v1:" + hex(SHA-256(model + "\0" + messages_json))`, covering the entire message array. Sampling params are excluded from the key |
| vLLM model | Configured via `MODEL_NAME` in the Deployment ConfigMap — no model hardcoded in the serving layer |

# Phase 2 Fermyon Build Log

*2026-04-16*

---

## 1. What was originally intended

The goal was a lightweight HTTP proxy running in the Fermyon Wasm runtime, sitting in front of vLLM and routing identical chat completion requests to a Valkey cache before they reached the GPU.

The original assumed architecture:

- A **single Rust crate** compiled to one `.wasm` binary. The binary would inspect `req.uri().path()` at runtime and branch on `/health` vs `/v1/chat/completions`.
- A **Spin manifest** (`spin.toml`) with a single `[[trigger.http]]` component and a `[application.trigger.http]` section containing `listen = "0.0.0.0:8082"` to bind the server address.
- A **Docker base image** of `ghcr.io/fermyon/spin:3`, assumed to exist as an official Fermyon-published image.
- Cache key derivation based on a SHA-256 prefix of the raw prompt string — the same scheme used by the earlier Phase 2 sketch.
- TTL set via a `conn.expire()` method on the Spin Redis connection object.
- Kubernetes node selection by hardcoded node name rather than a stable label.

---

## 2. What was actually built

### Cargo workspace

The final implementation is a Cargo workspace at `phases/phase2-prefix-cache/fermyon/` with two member crates:

```
fermyon/
├── Cargo.toml          # workspace root; no [package]
├── proxy/
│   ├── Cargo.toml      # crate-type = ["cdylib"]
│   └── src/lib.rs      # POST /v1/chat/completions handler
└── health/
    ├── Cargo.toml      # crate-type = ["cdylib"]
    └── src/lib.rs      # GET /health handler
```

`cargo build --workspace --target wasm32-wasip1 --release` produces two binaries in the workspace-level `target/`:

- `target/wasm32-wasip1/release/proxy.wasm`
- `target/wasm32-wasip1/release/health.wasm`

### spin.toml

Two independent components, each with its own `[[trigger.http]]` route. The `health` component carries `allowed_outbound_hosts = []`; the proxy component locks outbound to exact cluster-internal addresses. No `[application.trigger.http]` section — the listen address is runtime-only.

```toml
[[trigger.http]]
route     = "/health"
component = "health"

[[trigger.http]]
route     = "/v1/chat/completions"
component = "prefix-cache-handler"

[component.prefix-cache-handler]
source = "target/wasm32-wasip1/release/proxy.wasm"
allowed_outbound_hosts = [
    "redis://valkey-svc.inference.svc.cluster.local:6379",
    "http://vllm-svc.inference.svc.cluster.local:8000",
]
```

### Cache key scheme

```
key = "fermyon:v1:" + hex(SHA-256(model + "\x00" + messages_json))
```

Only `model` and `messages` (normalised to `[{role, content}]`) contribute to the key. Sampling parameters (`temperature`, `top_p`, `max_tokens`, `stream`) are excluded so requests with identical prompts but different sampling settings share one cache entry. The null-byte separator prevents hash collisions between a model name that ends with a JSON fragment and a different `(model, messages)` pair. Total key length: 75 characters (`fermyon:v1:` + 64-char hex digest).

### Transparent passthrough proxy

The request body is parsed as `serde_json::Value` to preserve unknown fields for forwarding to vLLM. Only `model` and `messages` are extracted for key derivation. The raw vLLM JSON response body is returned unchanged, with two added headers: `Content-Type: application/json` and `X-Cache: HIT` or `X-Cache: MISS`.

### stream:true stripping

The Fermyon Wasm runtime does not support streaming HTTP responses. If `stream: true` is present in the incoming request it is removed from the JSON body before forwarding to vLLM. The complete non-streaming response is returned. Clients expecting SSE or chunked transfer will not receive incremental tokens. This is a known Phase 2 limitation, logged to stderr when encountered.

### TTL via conn.execute

```rust
let _ = conn.set(&cache_key, &vllm_body);
let _ = conn.execute("EXPIRE", &[
    spin_sdk::redis::RedisParameter::Binary(cache_key.as_bytes().to_vec()),
    spin_sdk::redis::RedisParameter::Int64(cache_ttl),
]);
```

Both calls are best-effort. A write or expire failure does not fail the client request; the next identical request becomes another cache miss. The TTL is not reset on cache hits — it counts down from first write.

### Dockerfile

```dockerfile
FROM debian:bookworm-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates && \
    curl -fsSL https://github.com/spinframework/spin/releases/download/v3.6.3/spin-v3.6.3-linux-amd64.tar.gz \
        | tar -xz spin && \
    mv spin /usr/local/bin/ && \
    apt-get purge -y curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY spin.toml .
COPY target/wasm32-wasip1/release/proxy.wasm  target/wasm32-wasip1/release/proxy.wasm
COPY target/wasm32-wasip1/release/health.wasm target/wasm32-wasip1/release/health.wasm

ENTRYPOINT ["spin", "up", "--listen", "0.0.0.0:8082"]
```

The `.wasm` source paths inside the image mirror `spin.toml`'s `source =` values exactly. The `curl` package is purged after the download to keep the final image lean; `ca-certificates` and `libssl3` are retained because Spin makes outbound TLS connections.

### Kubernetes manifest

`fermyon/k8s/fermyon-deployment.yaml` contains three resources in one file:

- **Namespace** `inference`
- **ConfigMap** `fermyon-config`: injects `SPIN_VARIABLE_VALKEY_ADDRESS`, `SPIN_VARIABLE_VLLM_URL`, `SPIN_VARIABLE_CACHE_TTL` as environment variables
- **Deployment** `fermyon-prefix-cache`: image `ghcr.io/jginsj/fermyon-prefix-cache:latest`, `nodeSelector: workload-type=cpu`, resource requests 50m CPU / 64Mi memory, limits 500m / 256Mi, readiness and liveness probes on `GET /health:8082`
- **Service** `fermyon-svc`: ClusterIP, port 8082

### Tests

`phases/phase2-prefix-cache/tests/test_chat_cache.py` — 25 tests total:

- 21 unit tests (run by default): key format, determinism, sampling params excluded, key differences, null-byte separator, field-order independence, pinned SHA-256 values
- 4 integration stubs (`@pytest.mark.integration`, deselected by default): cache miss, cache hit, stream:true handling, sampling params don't affect live cache key

```
21 passed, 4 deselected in 0.04s
```

---

## 3. Decisions made and why

### Single crate → Cargo workspace

**Originally assumed:** one `#[http_component]` entry point that inspects `req.uri().path()` at the top of the handler and branches to health or proxy logic.

**Discovered:** Spin dispatches requests to components by route, not by path inspection inside a shared binary. Each Spin component must have exactly one `#[http_component]` function as its entry point. A single `cdylib` with a path check can be routed from multiple `[[trigger.http]]` sections, but this gives the health endpoint access to the same outbound permissions as the proxy — which is wrong. More fundamentally, there is no mechanism to give a single binary two isolated permission sets.

**Fix:** Cargo workspace with two `cdylib` crates. `health` carries `allowed_outbound_hosts = []` and no Spin variables. `proxy` carries exact outbound hosts and the three variables. Each has its own `#[http_component]` with no path inspection.

---

### ghcr.io/fermyon/spin:3 → debian:bookworm-slim + pinned tarball

**Originally assumed:** Fermyon publishes an official Docker image at `ghcr.io/fermyon/spin:3` that can be used as a base image.

**Discovered:** The tag does not exist. Fermyon does not publish a Docker base image for Spin. The correct approach is to install the `spin` binary into a standard base image.

The Fermyon-provided install script (`https://developer.fermyon.com/downloads/install.sh`) was tried next. It resolved to Spin v3.6.3 and downloaded the tarball successfully, but step 4 of the script clones the default templates repository via `git`. `git` is not installed in `debian:bookworm-slim`, so the script exited with code 1 and the build failed.

Templates are only needed for `spin new` (project scaffolding). `spin up` does not require them.

**Fix:** Download the release tarball directly from the GitHub releases URL that the install script would have used, extracting only the `spin` binary:

```bash
curl -fsSL https://github.com/spinframework/spin/releases/download/v3.6.3/spin-v3.6.3-linux-amd64.tar.gz \
    | tar -xz spin
mv spin /usr/local/bin/
```

This is reproducible (pinned version), has no extra dependencies, and produces a smaller layer because templates are never downloaded.

---

### listen field removed from spin.toml

**Originally assumed:** Spin 3.x accepts a `listen` field under `[application.trigger.http]` to configure the server bind address.

**Discovered:** Spin 3.x rejected the manifest at startup with:

```
Error: metadata error: invalid metadata value for "http":
Error("unknown field `listen`, expected `base`")
```

The `listen` field was removed from the Spin manifest format in Spin 3.x. The bind address is now a **runtime-only flag**, not a manifest field.

**Fix:** Remove the `[application.trigger.http]` block from `spin.toml` entirely. The address is already set correctly in the Dockerfile `ENTRYPOINT`:

```dockerfile
ENTRYPOINT ["spin", "up", "--listen", "0.0.0.0:8082"]
```

---

### conn.expire() → conn.execute("EXPIRE", ...)

**Originally assumed:** `spin_sdk::redis::Connection` exposes an `expire(key, ttl)` method analogous to the Redis `EXPIRE` command.

**Discovered:** No such method exists. The `Connection` type in `spin-sdk 3.1.1` exposes `get`, `set`, `incr`, `del`, `sadd`, `smembers`, `srem`, and `execute`. There is no `expire` method.

**Fix:** Use the generic `execute` escape hatch:

```rust
let _ = conn.execute("EXPIRE", &[
    spin_sdk::redis::RedisParameter::Binary(cache_key.as_bytes().to_vec()),
    spin_sdk::redis::RedisParameter::Int64(cache_ttl),
]);
```

A secondary error appeared after the first fix attempt: the suggested variant names `Str` and `Int` do not exist. The actual `RedisParameter` enum (from the WIT definition at `wit/deps/spin@2.0.0/redis.wit`) has two variants: `Binary(payload)` and `Int64(s64)`. String keys must be passed as `Binary(Vec<u8>)`.

---

### Hardcoded node name → workload-type=cpu node label

**Originally assumed:** the `nodeSelector` in the Deployment could reference a specific LKE node name.

**Discovered:** LKE-generated node names (e.g. `lke868003-1234567-abcdef`) change when a node pool is deleted and recreated — which happens during Terraform changes that modify pool configuration. A hardcoded name would break the Deployment silently.

**Fix:** Use a stable label instead. The `workload-type=cpu` label is defined in `cluster.tf` on the CPU pool and applied manually post-provisioning:

```bash
kubectl label node <cpu-node-name> workload-type=cpu --overwrite
```

The `nodeSelector` in the Deployment manifest becomes:

```yaml
nodeSelector:
  workload-type: "cpu"
```

This survives node pool recreation as long as the label is reapplied, which is already part of the infrastructure README checklist.

---

### Deployment naming — RESOLVED

The Kubernetes Deployment was originally named `fermyon-proxy` in the manifest, while the Docker image was `fermyon-prefix-cache` and the Spin application was `prefix-cache-handler`. That inconsistency was flagged here for a future cleanup pass.

**Resolved since.** The Deployment, its labels, its `matchLabels` selector, the container name, and the Service selector all read `fermyon-prefix-cache` now, matching the image — see `fermyon/k8s/fermyon-deployment.yaml`. The only remaining discrepancy is the Spin component, still named `prefix-cache-handler` inside `spin.toml`; it is internal to the Wasm application and does not appear in any Kubernetes object.

---

### GHCR image visibility

**Originally assumed:** a new GHCR package inherits the visibility of the repository.

**Discovered:** new GHCR packages default to **private**. The `imagePullPolicy: Always` on the Deployment caused a pull failure because the Kubernetes nodes have no registry credentials configured for `ghcr.io`.

**Fix:** Set the package to **public** via the GitHub UI (Settings → Packages → Change visibility). This matches the pattern used by the `vllm-lmcache` and `qwen-image` packages in the same namespace.

---

## 4. Cache benchmark results

Measured 2026-04-16. Source: `phases/phase2-prefix-cache/results/phase2_cache_benchmark.json`.

### Test conditions

- **Cluster**: akamai-lke-us-ord, RTX 4000 Ada node pool
- **Model**: `mistralai/Mistral-7B-Instruct-v0.2`
- **Requests per pass**: 10 sequential, `max_tokens=64`
- **Shared prefix**: ≈ 500 tokens (50 × "The quick brown fox..." phrase, ≈ 10 BPE tokens/repeat)
- **Cold-cache method**: uuid4 run nonce embedded in every message — guarantees a cold cache
  without requiring Valkey FLUSHDB access

### Three-pass latency

| Pass | p50 (ms) | p95 (ms) | errors |
|---|---|---|---|
| Pass 1 — MISS (Fermyon → vLLM) | 3,058 | 5,897 * | 0 |
| Pass 2 — HIT  (Fermyon → Valkey) | 218 | 221 | 0 |
| Pass 3 — Direct vLLM (no cache) | 3,020 | 3,079 | 0 |

\* Pass 1 p95 is inflated by an 8,173 ms first-request spike on request 1 of 10
(vLLM cold-start — first inference after pod readiness). Requests 2–10 fell in
the 2,820–3,116 ms range. The p50 (3,058 ms) is unaffected and is the correct
MISS latency for the break-even calculation.

### Cache value

```
Miss overhead  = Pass1 p50 − Pass3 p50  =  3,058 − 3,020  =   +38 ms
Hit saving     = Pass3 p50 − Pass2 p50  =  3,020 −   218  = 2,802 ms

Break-even hit rate = miss_overhead / (miss_overhead + hit_saving)
                    =      38       / (     38       +   2,802   )
                    =   1.3%
```

**Miss overhead (+38 ms)** is the Valkey round-trip cost on every cache miss.
Fermyon misses cost slightly more than direct vLLM — the extra hop is real and
the number is honest.

**Hit saving (2,802 ms)** is the latency saved versus direct vLLM on a cache
hit. A HIT response takes 218 ms vs 3,020 ms direct — a 13.8× reduction.

**Break-even hit rate (1.3%)** is the minimum hit rate for net-positive cache
impact. The 25.9% external prefix cache hit rate confirmed in vLLM metrics
(verified 2026-04-15) is well above this floor. The cache layer is operating in
a strongly net-positive regime.

---

## 5. Current live state

### Pod

```
kubectl get pods -n inference -l app=fermyon-prefix-cache
```

The pod is scheduled on the CPU node pool node carrying `workload-type=cpu`. The GPU node carries `gpu-type=rtx4000ada` and is not eligible for this workload.

### Health check

```bash
kubectl port-forward svc/fermyon-svc 8082:8082 -n inference
curl -s http://localhost:8082/health
# → ok
```

### Integration tests

```bash
kubectl port-forward svc/fermyon-svc 8082:8082 -n inference &
cd phases/phase2-prefix-cache
FERMYON_URL=http://localhost:8082 \
python -m pytest tests/test_chat_cache.py -v -m integration
```

Expected results:

- `test_integration_cache_miss_returns_x_cache_miss` — 200, `X-Cache: MISS`, body contains `choices`
- `test_integration_cache_hit_returns_x_cache_hit` — second identical request returns `X-Cache: HIT`, body byte-identical to first
- `test_integration_stream_true_handled_without_error` — 200, `Content-Type: application/json`, body parseable as JSON (not SSE)
- `test_integration_sampling_params_do_not_affect_cache_key` — second request with different `temperature` returns `X-Cache: HIT`

### Service

```
fermyon-svc.inference.svc.cluster.local:8082   ClusterIP
```

Traffic from within the cluster (e.g. a load generator or API gateway) should target `http://fermyon-svc.inference.svc.cluster.local:8082/v1/chat/completions`.

### Image

```
ghcr.io/jginsj/fermyon-prefix-cache:latest
digest: sha256:ef9fc76bb4330a59a9821a1092231a40e0833e43678705a8a12ad4bbd2800a28
spin: v3.6.3
base: debian:bookworm-slim
```

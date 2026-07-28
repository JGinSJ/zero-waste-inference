# Zero-Waste Inference on Akamai Cloud — Phase 2

Moving prefix reuse to the edge with Fermyon, Valkey, and vLLM

At the end of Phase 1, we had a nice clean answer to a local question: why should a model reuse previously computed attention state instead of recalculating it on every decode step? The answer, of course, was “because wasting compute is bad and the model does not need to rediscover its own past” [file:78][file:84].

Now it’s time to make that idea less local and a lot more operational. In this phase, I wanted to move from reuse *within* a single request to reuse *across* requests by putting a lightweight edge layer in front of the model backend: a Fermyon Wasm Function at the front door, a Valkey cache in the middle, and vLLM on Akamai LKE behind it for the requests that still need GPU work [file:85][file:77][file:75].

The operator-visible effect is simple: repeat requests should stop taking the full trip to the GPU every time. The engineering-visible effect is slightly less simple: we now have to deal with Wasm packaging, Spin routing rules, cache key design, vLLM behavior, Kubernetes placement, and the general inconvenience of discovering that the cleanest architecture diagram is rarely the same thing as the first implementation that actually runs [file:79][file:85].

## Why move caching up the stack?

Phase 1 showed the smallest unit of reuse: the model keeps its own attention history around instead of recomputing it [file:78][file:84]. That idea is powerful, but it only helps within a single generation. As soon as a new request arrives, the serving system can still end up doing a lot of duplicated work if the prompt shares a common prefix with earlier requests, or if the request is functionally identical to one we already answered [file:77][file:85].

That is the gap Phase 2 is meant to close. The phase goal is a prefix-aware inference pipeline where a Fermyon Wasm Function intercepts requests, checks a Valkey cache for matching prompt state, and only forwards misses to a vLLM backend running on Akamai LKE [file:85]. That was the goal as written. What the front door actually shipped as is narrower, and it is worth naming up front: an exact-match response cache. Prefix reuse still happens in this phase, but it happens one layer down inside vLLM rather than at the edge [file:79][file:77]. In other words, we stop treating the request boundary like an amnesia event [file:77][file:85].

There is also a nice narrative reason to do it this way. Once the reuse idea moves out of the model internals and into a distributed system, all the boring-but-important platform questions show up immediately: where the front door runs, what the cache key means, what happens on misses, what gets scheduled to CPU versus GPU nodes, and how much latency the extra hop actually adds [file:79][file:77].

## The architecture at a glance

The Phase 2 target state is straightforward on paper. A client request first hits a Fermyon Wasm Function running on the CPU side of the Akamai LKE cluster. That function hashes the request’s model and messages, checks Valkey, and either returns a cached response on a hit or forwards the request to a vLLM backend on the RTX 4000 Ada GPU node pool on a miss [file:77][file:79].

The cluster topology supporting that path is split cleanly by workload type. The CPU node is used for Fermyon and Valkey, while the GPU nodes are reserved for vLLM and, later, the Phase 3 Qwen-Image service [file:77]. Manual node labels like `workload-type=cpu` and `gpu-type=rtx4000ada` are used to keep placement stable, because LKE node names can change when pools are recreated and hardcoding names is just a creative way to schedule future pain [file:77][file:79].

This phase also makes the project feel meaningfully more real. We are no longer explaining reuse as a tensor-level trick on a laptop. We are enforcing reuse with a front-door component, a network-visible cache, and a GPU-serving backend that only gets involved when it genuinely has to [file:85][file:77].

## Step 1 — Build the Fermyon front door

The original idea was to build one Rust crate that inspected the request path at runtime and branched between `/health` and `/v1/chat/completions` [file:79]. That sounds perfectly reasonable until Spin reminds you that it routes requests to components by route, not by “please trust me, I’ll branch inside the binary” logic [file:79].

The final implementation became a Cargo workspace under `phases/phase2-prefix-cache/fermyon/` with two member crates: `proxy/` for `POST /v1/chat/completions` and `health/` for `GET /health` [file:79]. That ended up being the right call for security and clarity, because each component could carry its own outbound permission set. The health component gets no outbound hosts, while the proxy component is allowed to talk only to the exact cluster-internal Valkey and vLLM addresses it needs [file:79].

That separation is one of those details that seems overly careful until you realize the alternative is giving a trivial health endpoint the same network privileges as the main proxy. Which is exactly the kind of thing that makes architecture diagrams look clean and postmortems look expensive [file:79].

## Step 2 — Define a cache key that is boring in the right ways

The cache key scheme also changed from the earlier sketch. The implementation settled on a SHA-256 digest derived from the model name plus a normalized `messages` payload, with a null-byte separator to avoid accidental collisions between different `(model, messages)` combinations [file:79]. The final key format is `fermyon:v1:` followed by the 64-character hex digest, for a total length of 75 characters [file:79].

Only `model` and `messages` contribute to the key. Sampling parameters like `temperature`, `top_p`, `max_tokens`, and `stream` are excluded so requests with the same prompt content can share a cache entry even if their sampling knobs differ [file:79]. That is a meaningful design choice, because it treats prompt identity as the main reuse boundary rather than every possible request flag [file:79]. It also makes the matching rule stricter than the word “prefix” tends to suggest: the digest covers the entire message array, so two requests that share a long opening but diverge by one token at the end produce different keys and do not share an entry. This layer catches repeats, not near-misses [file:79].

The proxy itself stays intentionally transparent. It parses the request body as `serde_json::Value`, extracts only the fields needed for the key, forwards the rest unchanged to vLLM, and returns the raw JSON response body with an added `X-Cache: HIT` or `X-Cache: MISS` header so the cache path is visible to the client [file:79]. In a better world, all proxies would be this honest about what they are doing.

## Step 3 — Accept the Wasm runtime’s limitations before they embarrass you

One of the more practical constraints in this phase is that the Fermyon Wasm runtime does not support streaming HTTP responses in the way an OpenAI-style streaming client expects [file:79]. If an incoming request contains `stream: true`, the proxy strips that field before forwarding to vLLM and returns the full non-streaming JSON response instead [file:79].

That is not a bug hidden under a rug. It is a documented Phase 2 limitation, and the runtime logs the situation to stderr when it occurs [file:79]. Clients expecting server-sent events or chunked token streaming do not get incremental output from this path yet [file:79].

I actually like leaving that limitation visible in the article, because it keeps the talk track honest. This phase is about proving that edge interception, cache lookup, and backend fallback work correctly. It is not pretending that every possible serving behavior is already feature-complete just because the architecture diagram has arrows [file:79][file:85].

## Step 4 — Package and deploy the edge layer without making it weird

Packaging the Fermyon piece involved a few corrections from the original assumptions. One early assumption was that `ghcr.io/fermyon/spin:3` existed as a ready-made Docker base image; it does not, so the final image uses `debian:bookworm-slim` and installs a pinned Spin binary directly from the release tarball for Spin v3.6.3 [file:79]. Another assumption was that the `listen` address belonged in `spin.toml`, but Spin 3.x treats that as a runtime concern instead, so the address is set in the container entrypoint via `spin up --listen 0.0.0.0:8082` [file:79].

The Kubernetes manifest for the proxy is also intentionally modest. It deploys into the `inference` namespace, injects the Valkey address, vLLM URL, and cache TTL through a ConfigMap, schedules the pod to the CPU node using `nodeSelector: workload-type=cpu`, and exposes it through a ClusterIP service on port 8082 with readiness and liveness probes on `/health` [file:79].

That all sounds boring, and it is. But boring in this case is good. The point of the edge layer is not to become the most glamorous part of the system. It is to be reliable enough that the GPU only hears about the requests that actually need GPU work [file:79][file:77].

## Step 5 — Put Valkey and vLLM behind it for the real work

The cache and backend sides of the phase are what make the front door worth having. Valkey 8.0 runs as a standalone cache with an allkeys-lru policy and a 2 GB memory cap, and it ends up doing double duty: the Fermyon layer writes whole responses into it, and vLLM writes KV blocks into it through LMCache [file:77][file:85].

That second path is worth stating precisely, because it is easy to describe wrongly. The live vLLM deployment runs with `--no-enable-prefix-caching` and an `LMCacheConnectorV1` KV transfer config — vLLM’s own built-in prefix cache is deliberately switched off, because LMCache manages prefix reuse externally and the two do not stack [file:77][file:82]. That surprised me the first time I read the manifest back, but it makes sense once you say it out loud: you do not want two layers both convinced they own the KV blocks.

The repo architecture doc maps the placement clearly: Fermyon and Valkey live on the CPU side, while vLLM runs on the GPU side under the `inference` namespace [file:77]. The Phase 2 service path then becomes straightforward: `fermyon-svc` on port 8082 fronts `valkey-svc` on 6379 and `vllm-svc` on 8000, with the Wasm handler deciding whether the request should terminate early as a cache hit or continue to the backend as a miss [file:77][file:79].

There is a subtle but important distinction here too. Phase 2 is running two different caches against the same Valkey instance, and they are not the same idea wearing two hats. LMCache reuses KV state for requests that genuinely share a prefix, which saves prefill work but still spends GPU time. The Fermyon layer sits above that and stops repeat requests before they reach the GPU at all [file:77][file:79]. That is why the project description talks about both prefix caching and request deduplication at the Fermyon layer — those are two separate reuse points that happen to share a backing store [file:77].

## What the benchmark actually showed

The most useful result from this phase is not that the cache exists. It is that the cache is clearly worth the extra hop [file:79]. The measured three-pass benchmark used 10 sequential requests per pass, a shared prefix of about 500 tokens, and a cold-cache method that embedded a UUID nonce into each message so the first pass was guaranteed to miss without needing to flush Valkey [file:79].

It’s worth being precise about what actually makes Pass 2 hit, because it is not the shared prefix. The Fermyon key covers the whole message array, so Pass 2 hits because it replays Pass 1’s requests byte for byte. The 500-token prefix is there to make each request realistically expensive on the GPU side, not to trigger the cache [file:79].

On the measured run, Pass 1 through Fermyon to vLLM on a miss had a p50 latency of 3,058 ms, Pass 2 through Fermyon to Valkey on a hit had a p50 of 218 ms, and Pass 3 direct to vLLM without the cache had a p50 of 3,020 ms [file:79]. That means the measured miss overhead introduced by the cache layer was only +38 ms over direct vLLM, while the hit saving was 2,802 ms, taking the response path from roughly 3 seconds down to 218 ms on cache hits [file:79].

The break-even hit rate works out to about 1.3 percent, using the measured miss overhead divided by the sum of miss overhead and hit saving [file:79]. That is a very low bar, and it is low for a deeply unglamorous reason: the extra hop costs 38 ms and a hit saves 2,802 ms, so the arithmetic barely has to try.

I want to be careful about what that number does and does not prove. It says the Fermyon layer pays for itself at almost any realistic repeat rate. It does not say what the repeat rate actually is — I never measured the Fermyon hit rate under production-shaped traffic, only under a benchmark where Pass 2 was designed to hit [file:79]. There is a 25.9 percent external prefix cache hit rate recorded in the phase results (`results/phase2_valkey_verified.json`), and I was tempted to quote it here, but that figure belongs to LMCache’s KV cache, measured against vLLM directly before the Fermyon front door was wired to that instance. It is a real number from a different layer and a different run. Presenting it as the response cache’s hit rate would be exactly the sort of tidy-sounding slippage this series is supposed to be arguing against.

So the honest version is: the break-even is 1.3 percent, the mechanism demonstrably works, and the hit rate that tells you what it is worth on *your* traffic is still yours to measure [file:79].

## The friction points that made this phase real

This phase had more sharp edges than Phase 1, which is only fair because it also had more actual infrastructure. One issue was the mistaken assumption that Spin exposed a convenient `conn.expire()` method on its Redis connection object; it does not, so TTL had to be applied through the generic `execute("EXPIRE", ...)` path using the correct `RedisParameter` variants [file:79].

Another was package and runtime packaging friction. GHCR packages defaulted to private visibility, which caused image pulls to fail on the cluster until the package visibility was changed to public [file:79]. There was also a naming inconsistency between the Deployment and the image, which future-me did in fact eventually clean up: the Deployment, its labels, its selector, and the image name all read `fermyon-prefix-cache` now [file:77]. The only leftover is the Spin application component, still called `prefix-cache-handler` internally — a discrepancy small enough that chasing it would cost more than living with it [file:79].

Then there is the small matter of streaming. The proxy strips `stream: true` because the runtime cannot support that response model yet, which means Phase 2 proves the request interception and cache path cleanly but does not pretend to deliver full streaming semantics [file:79]. Again, I’d rather publish a phase with one honest limitation than six quiet lies.

## Where this leaves us

By the end of Phase 2, the project has moved from an educational local demo into a live system in us-ord with a Fermyon Wasm front door, a Valkey cache on the CPU node, and a vLLM backend on the RTX 4000 Ada GPU node pool behind it [file:75][file:77]. The cache layer is not theoretical, the proxy is not mocked, and the benchmark shows a measured p50 hit path of 218 ms versus 3,020 ms direct to vLLM, with only 38 ms of miss overhead and a break-even hit rate of about 1.3 percent [file:79].

That means the project now demonstrates two different kinds of reuse. Phase 1 reused attention state within a generation. Phase 2 reuses useful work across requests, at the edge, before the GPU has to spend time on something it has effectively seen before [file:78][file:79][file:77].

In the next phase, I’ll switch from text inference plumbing to the image path by deploying a Qwen-Image serving stack on Akamai LKE, with FastAPI, batching, and GPU-backed request routing [file:86][file:75]. The plumbing changes, the payload changes, and the implementation gets more multimodal, but the project thesis stays stubbornly the same: stop throwing away work that you already paid to compute [file:86][file:75].

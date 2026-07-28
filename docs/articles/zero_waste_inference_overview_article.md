# Zero-Waste Inference on Akamai Cloud

How to stop throwing away useful compute during inference

*The overview. Three parts follow. The [project repo](https://github.com/JGinSJ/zero-waste-inference) is organised into four phases, which do not map one-to-one onto the parts — more on that at the end.*

In theory, modern AI inference should get cheaper as the software gets smarter. In reality, a lot of inference systems still behave like they have never heard of the word “reuse.” We compute the same attention state over and over, reprocess the same prompt prefixes across requests, and then wonder why the GPU bill keeps showing up with bad news.

This project started from a simple idea: if inference is expensive because useful work is constantly discarded, then a good demo should show exactly where that waste happens and what changes when you stop throwing that work away. So in this series, I’m building a phased, production-quality demo on Akamai Cloud that reduces inference cost by reusing computation at several different layers of the stack, from a tiny PyTorch transformer all the way up to edge caching, GPU-backed serving, and cost modeling on Akamai LKE.

If that sounds abstract, here’s the less abstract version: I wanted something I could actually build, run, benchmark, explain, and publish without hand-waving over the painful parts. Which means this series is not just about models. It’s about the plumbing that makes reuse possible in the first place.

## The problem: inference waste hides in plain sight

Most of the public conversation around inference cost focuses on model size, quantization, or which GPU you rented. Those things matter, of course, but they can also distract from a more boring and more important question: how much work are you recomputing that you didn’t need to recompute at all?

That waste shows up in a few places. Within a single autoregressive generation, you can avoid recomputing attention history by keeping a key-value cache. Across requests, you can avoid recomputing shared prompt prefixes by keeping the computed KV state somewhere both cheap and nearby. At the service layer, you can short-circuit repeated requests entirely, before the model backend is involved at all. And at the deployment layer, you can measure how concurrency changes throughput and cost so you stop paying premium prices for badly utilized GPUs.

Those are four different mechanisms, and it is worth being precise that they are different — a lot of writing on this topic blurs them into one word, “caching,” and then the reader cannot tell which one is doing the work. This series keeps them apart on purpose.

That is the through-theme for the project: not “AI is expensive,” but “AI is often wasteful in very specific, fixable ways”. Sorry, I couldn’t come up with anything more poetic than that.

## What this project actually builds

The repo is organized into four phases, each focused on a different reuse mechanism in the inference path. Phase 1 builds a minimal PyTorch transformer from scratch so the KV cache is visible, measurable, and impossible to treat like magic. Phase 2 moves up the stack and puts a Fermyon Wasm Function in front of a Valkey cache and a vLLM backend on Akamai LKE, so repeat requests are answered without waking the GPU and shared prefixes are reused when the GPU does get involved.

Phase 3 shifts to image-model serving on Akamai LKE using a Qwen2.5-VL serving path, with FastAPI and explicit GPU deployment concerns rather than a hand-wavy “imagine this is production” diagram. Phase 4 then measures what the GPU layer is actually costing by running a benchmark harness against an RTX 4000 Ada deployment, converting throughput into cost-per-token and cost-per-million-tokens so the system can be discussed in operational terms, not just architectural ones.

Here’s the project in one sentence: show reuse within a request, across requests, in front of requests, and then measure what that means once a real GPU-backed service is running.

## The architecture at a glance. For now.

At the center of the running system is an Akamai LKE cluster in the us-ord region, with a CPU node handling the Phase 2 support services and RTX 4000 Ada GPU nodes handling the model-serving workloads. In the current topology, Fermyon and Valkey live on the CPU side, while vLLM and the Qwen serving path live on the GPU side, all under an `inference` namespace with explicit node labeling to keep scheduling predictable.

The Phase 2 request flow is the clearest example of the project’s intent, and it is worth stating precisely because it is easy to describe wrongly. A client request first hits a Fermyon Wasm Function, which hashes the request’s **model and messages** — the whole message array, not a prompt prefix — and checks Valkey. An exact repeat comes back in about 218 ms without the GPU doing anything at all; everything else falls through to the vLLM backend on the GPU node pool, where LMCache handles prefix-level reuse against that same Valkey instance.

So there are two caches sharing one Valkey, and they are not the same idea wearing two hats. The front door catches *repeats*. LMCache catches *shared prefixes*, which still cost GPU time but less of it. Getting those two confused is the single easiest way to overstate what a cache layer is doing for you, and I managed to do exactly that in an early draft of this very article.

The measured version of that story: a cache hit returns in 218 ms against 3,020 ms straight to the GPU, while the extra hop costs a miss only 38 ms. That is the number the rest of the series is trying to earn.

By Phase 4, the same infrastructure is useful not just for serving, but for measuring throughput and cost under increasing concurrency, so the “does this save money?” question has an answer grounded in runs instead of vibes.

## Why Akamai Cloud is the testbed

I chose Akamai Cloud because the project needed a place where the edge, the cluster, and the GPU-backed workloads could all be part of one coherent story rather than four disconnected demos duct-taped together for a slide deck. Akamai LKE is the Kubernetes target throughout, and the repo’s live phases are already tied to deployments in us-ord for the caching and Qwen paths.

That matters because this series is not trying to prove that a single optimization trick exists in isolation. It is trying to show that reuse can be made visible end to end: from a developer laptop, to request interception at the front door, to GPU-hosted serving, and finally to measured cost behavior.

Or, put another way, the point is not merely to say “KV cache good.” The point is to demonstrate how several reuse ideas can be staged across a real deployment path until they become something an engineer could clone, run, and reason about without needing a séance.

## The part where reality interrupts the plan

Originally, Phase 4 was intended as a multi-GPU comparison between RTX 4000 Ada and RTX PRO 6000 Blackwell using a tensor-parallel vLLM deployment and a shared cost model. That is the kind of thing that sounds excellent in a planning doc and considerably less excellent when the Blackwell node pool is not on your cluster and the tensor-parallel target needs two GPUs on one node.

So Phase 4 was reframed into what was actually measurable: a single-GPU RTX 4000 Ada baseline, using Mistral-7B-Instruct-v0.2 on Akamai LKE, with a synchronous concurrency sweep across 1, 2, 4, 8 and 16 and a cost model based on a confirmed GPU hourly price of $0.96 for the Ada node used in the run. That produced a real result set instead of a speculative comparison: throughput rose from about 20.9 tokens per second at concurrency 1 to 184.8 at concurrency 16, while cost per million output tokens fell from $12.77 to $1.44 at the prompt≈256-token setting.

Honestly, I like this version better. Not because I no longer want the Blackwell comparison — the node type has since appeared on the platform, and the Terraform stub is sitting there waiting — but because a real Ada-only baseline is more useful than an imaginary head-to-head chart that never left the whiteboard.

## Plumbing vs payload, again

If you’ve read my other technical series, you probably know I have a bad habit of obsessing over the plumbing. I am happy to report that this project has done nothing to cure that condition.

The payload here is the inference result, whether that is a generated token stream, a cached response, or an image-model reply. The plumbing is everything that makes the result efficient and repeatable: the explicit KV cache logic, the Fermyon ingress path, the Valkey cache, the vLLM and LMCache configuration, the Akamai LKE node placement, the FastAPI serving layer, the benchmark harness, the cost model, and the uncomfortable but necessary distinction between what was planned and what was actually measured.

Most demos jump straight to the sexy part and quietly skip the parts where architecture decisions, cache boundaries, cluster scheduling, and benchmark hygiene decide whether the system is credible. In this series, those “boring” pieces are first-class, because they are the difference between a demo that sounds sharp and a demo that survives contact with reality.

## What the rest of the series covers

Three articles follow this one.

**Part 1 — Building the KV cache from scratch.** A minimal PyTorch transformer that makes the KV cache visible enough to inspect, test, and time, without relying on HuggingFace or any other inference framework to hide the mechanism. It is the foundation for everything else, because it shows exactly what is being reused inside a single generation, and it ends with a measured 3.69× decode speedup and a test suite that checks the cached and uncached paths produce identical logits — not just identical tokens.

**Part 2 — Caching at the front door.** The Phase 2 system on Akamai LKE: Fermyon Wasm Functions intercepting requests, Valkey as the cache, and vLLM behind it with LMCache managing prefix reuse. This is where the 218 ms number comes from, along with an honest accounting of what the benchmark does and does not prove.

**Part 3 — What the GPU actually costs.** The measured RTX 4000 Ada benchmark and cost model, with the Blackwell path clearly labeled as planned but not yet benchmarked.

You may have noticed that is three parts against four phases. That is deliberate. **Phase 3 — the Qwen image-serving path — is built, deployed, and documented in the repo, but it is not in this series**. It tells the same reuse story with a different payload, and I would rather publish three articles that each earn their place than four where one is there to complete a numbering scheme. If you want it, it is in the repo with its build log.

So the repo keeps phases 1 through 4, the series runs Parts 1 through 3, and I will say “Phase N” when I mean the repo’s work and “Part N” when I mean the reading order.

By the end, the goal is something more useful than a pile of disconnected optimizations. The goal is a reusable blueprint for reducing inference waste on Akamai Cloud by reusing work at the right layer, measuring what changed, and being honest about what is still pending.

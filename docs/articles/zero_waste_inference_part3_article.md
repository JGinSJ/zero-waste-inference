# Zero-Waste Inference, Part 3 — What the GPU actually costs

Benchmarking the hardware you have, not the one your planning doc wanted

*Part 3 of three. This is Phase 4 in the [project repo](https://github.com/JGinSJ/zero-waste-inference) — Phase 3 is the Qwen image path, which is built and documented but sits outside this series. Repo phase numbers and series part numbers are not the same thing, and the article below refers to “Phase 4” whenever it means the repo’s work.*

Up to this point in the series, the project has been about building reuse into the inference path: KV caching inside the model in Part 1, then an exact-match response cache at the front door and LMCache-backed prefix reuse behind it in Part 2. But eventually every optimization project runs into a more practical question than “is this clever?” The question is: what does the system cost to run once the GPU is in the loop?

That is what Phase 4 is for. The original plan was a multi-GPU benchmark and cost model comparing RTX 4000 Ada against RTX PRO 6000 Blackwell on Akamai Cloud, using a tensor-parallel vLLM deployment and a benchmark harness that could measure throughput and latency across GPU tiers. What actually got built, however, was narrower and more honest: a single-GPU RTX 4000 Ada baseline benchmark on the hardware that was actually available, paired with a cost model that converts measured throughput into cost per token and cost per million tokens.

And frankly, I think that is the better article. Not because the Blackwell comparison stopped being interesting, but because real numbers on real hardware beat hypothetical numbers on unavailable hardware every time.

## The original plan versus the thing that shipped

On paper, Phase 4 was supposed to be the big comparative ending. The documented goal was a reproducible head-to-head throughput and cost comparison between RTX 4000 Ada and RTX PRO 6000 Blackwell, plus guidance for choosing the right GPU tier and parallelism strategy for a workload. The original architecture assumed an async load generator using `aiohttp` and SSE streaming, a tensor-parallel vLLM deployment with `--tensor-parallel-size 2`, and a CLI-driven harness sweeping across batch sizes and concurrency levels.

Then reality showed up. The RTX PRO 6000 Blackwell node pool was not yet available on the cluster at implementation time, and the tensor-parallel target required two GPUs on one node, which the existing Ada pool did not provide. That forced a scope correction: Phase 4 was reframed as a single-GPU Ada benchmark measuring raw decode throughput and output-token cost without any external cache in the loop.

That scope correction matters because it keeps the benchmark interpretable. The Phase 2 prefix-cache hit rate is intentionally excluded from Phase 4, since mixing cache wins into the GPU baseline would make the per-token cost model harder to reason about and less useful as a clean reference point. Sometimes the disciplined choice is to measure fewer things so the one thing you *do* measure actually means something.

## Why a baseline benchmark still matters

There is a temptation, especially in infrastructure-heavy projects, to treat “baseline” as a disappointing word. But in practice, a good baseline is often more useful than a half-finished comparison. The README now reflects that reality clearly: Phase 4 is listed as a single-GPU benchmarking and cost-model phase with RTX 4000 Ada measured and Blackwell pending.

That is exactly the right framing. A reproducible benchmark on the GPU tier you actually ran tells you how concurrency affects throughput, how latency behaves under load, and what your output tokens really cost at a known hourly node price. A future Blackwell comparison can still be added later, but it will be better because it will compare against a grounded Ada baseline instead of against wishful thinking.

There is also a narrative reason this works well at the end of the series. The earlier parts all argued that reuse reduces waste. This one closes the loop by asking what that waste looks like in money and throughput terms once a real deployment is sitting on a billable GPU.

## The benchmark harness that actually ran

The benchmark implementation is intentionally simpler than the original plan. Instead of an async SSE-based harness, the built version uses a synchronous sweep script in `benchmark/benchmark.py` with `requests` and `concurrent.futures.ThreadPoolExecutor`, targeting non-streaming `/v1/completions` only. No `aiohttp`, no `asyncio`, no SSE timing logic, and no TTFT/ITL breakdown in the final measured run.

That is not a step backward so much as a cleanup of ambition to match available hardware and a concrete benchmark target. The script takes CLI flags for the endpoint URL, model ID, approximate prompt length, output length, concurrency, request count, GPU hourly cost, output CSV path, and an optional run tag, then appends one row per run without overwriting prior results. That incremental append behavior is especially useful because it lets the sweep be rerun selectively without invalidating the whole result set.

Prompt construction is also intentionally explicit. The script repeats “The quick brown fox jumps over the lazy dog.” until the approximate prompt token target is reached, using the rough fact that the phrase tokenizes to about 10 BPE tokens for the relevant vocabularies. It is a deliberately approximate method, but an honest one, and the build log explicitly notes that character-count heuristics were considered and rejected because they do not track token boundaries reliably.

## The cost model is simple on purpose

One of the cleaner design choices in this phase is the cost model itself. Rather than burying pricing logic in the harness, `harness/cost_model.py` exists as a library and is called directly by the benchmark script on every run. The formulas are straightforward: cost per token is `gpu_hourly_usd / 3600 / tokens_per_second`, cost per request is that token cost multiplied by the mean output tokens generated, and cost per million tokens is simply cost per token multiplied by one million.

That simplicity is a feature. It makes it obvious what the benchmark is and is not claiming. The build log is explicit that these are **output-token costs** at varying concurrency for the measured Ada node, not a cross-provider comparison and not a complete total-cost-of-ownership exercise.

For the measured run, the Ada node price used in `configs/rtx4000ada.yaml` was $0.96 per hour for the `g2-gpu-rtx4000a1-l` node. Once that price is combined with measured throughput, the benchmark stops being an abstract performance exercise and becomes something you can reason about operationally: how much useful output you are getting for each GPU hour you are paying for.

## The measured setup

The benchmark ran on Akamai LKE in the `us-ord` region, using a single RTX 4000 Ada node with 20 GB of VRAM. The measured model was `mistralai/Mistral-7B-Instruct-v0.2`, served in float16 with `max_model_len=25664` and prefix caching disabled. Prefix caching was turned off on purpose in the Phase 4 deployment because the live Phase 2 setup already used LMCache-based prefix reuse, and leaving another caching layer enabled here would have polluted the clean GPU baseline.

The deployment used `k8s/vllm-ada.yaml` with `nodeSelector: gpu-type=rtx4000ada` and a single `nvidia.com/gpu: 1` allocation, then exposed the service locally through a port-forward to `svc/vllm-ada` on port 8000. Health was checked with a simple `/health` probe returning HTTP 200 before the sweep began.

The sweep itself covered concurrency levels 1, 2, 4, 8, and 16, approximate prompt lengths of 128, 256, and 512 tokens, fixed `max_tokens=64`, 10 requests per run, and a single warm-up request before the full matrix. That produced 15 benchmark runs plus the warm-up, all appended into `results/phase4_raw_benchmark.csv`, with zero recorded errors.

## What the benchmark showed about latency

The first headline from the results is that latency stayed surprisingly flat as concurrency increased. For the measured p95 end-to-end latency, the run moved from about 3,101 ms at concurrency 1 to about 3,490 ms at concurrency 16, which is only a 13 percent increase while concurrency grew by a factor of 16. The build log calls out that there was no real cliff in this range, just modest movement even at the higher prompt lengths.

That is an important result because it suggests the vLLM deployment was batching concurrent requests effectively instead of simply queueing them one behind another. In other words, increasing concurrency did not immediately destroy the latency profile; it mostly helped fill the decode batch.

For a client targeting less than 4,000 ms p95 end-to-end latency, the benchmark indicates that concurrency 16 remained viable on this setup. That is exactly the kind of “not glamorous, but very useful” information that a real deployment decision needs.

## What the benchmark showed about throughput

The second headline is throughput. Aggregate output throughput increased from about 21 tokens per second at concurrency 1 to about 185 tokens per second at concurrency 16 for the p=128 case, with similarly strong scaling across the other prompt lengths. The build log characterizes this as near-linear growth, and the numbers support that description: the 185.4 tok/s at c=16 for p=128 is about 8.8 times the 21.0 tok/s observed at c=1.

The README’s summary row for the prompt≈256-token case tells the same story in a compact way: 20.9 tok/s at concurrency 1, 67.5 at 4, 92.4 at 8, and 184.8 at 16. Whatever else you want to say about the hardware, the useful thing here is that the Ada node was not saturated at c=16.

That matters because it means the benchmark found an improving operating region, not a wall. The build log explicitly notes that follow-up sweeps at c=32 and c=64 would be needed to find the saturation ceiling where adding more requests no longer increases throughput because the decode batch becomes memory-bandwidth-bound. That follow-up was not part of the measured Phase 4 results, but the absence of a plateau at c=16 is itself a meaningful outcome.

## What the benchmark showed about cost

The third headline is the one that makes Phase 4 worth having at all: cost per useful output dropped dramatically as concurrency increased. For the measured prompt≈256-token case in the README, the cost per million output tokens fell from $12.77 at concurrency 1 to $3.95 at 4, $2.89 at 8, and $1.44 at 16. The build log broadens that picture across prompt lengths and shows the same trend consistently, with c=16 landing in a narrow $1.44 to $1.45 per million output-token range across prompt sizes.

That is an 8.8× cost improvement from the low-concurrency case to the highest measured concurrency, tracking throughput almost directly because the hourly GPU price stays fixed while the useful output rate rises. The practical lesson is simple: on this measured setup, higher concurrency was the cheaper operating point as long as the latency budget stayed acceptable.

It is also notable that prompt length had very little effect on cost at c=16 in the measured runs. That stability suggests decode throughput, rather than prompt-length variation in this range, dominated the cost behavior once the GPU was kept busy enough.

## The testing story matters too

One thing I like about this phase is that the harness was not treated as disposable glue code. The test suite covers it with 27 benchmark tests and 30 cost-model tests, for 57 passing tests total without requiring a GPU or network connection. Coverage included prompt scaling, CSV append behavior, header handling, concurrency ceilings, error capture, and cost arithmetic based on `usage.completion_tokens` from the response body.

That matters more than it might seem. Once you start turning benchmark numbers into cost claims, the harness itself becomes part of the argument. If the measurement and CSV-writing path are sloppy, the resulting “insights” are just expensive formatting.

It also leaves the phase in a better state for future expansion. A Blackwell run, if and when the node pool becomes available, will have a harness and cost-model foundation that already behaves predictably instead of needing to be reinvented during the comparison phase.

## The unfinished part is still part of the story

Phase 4 is complete in the repo’s current framing, but it is also openly unfinished in one specific way: the Blackwell comparison remains staged rather than measured. The architecture and scope docs still preserve the original multi-GPU intent, including `k8s/vllm-tp.yaml`, a `configs/rtxpro6000.yaml` placeholder, and open questions around node pricing, node labels, interconnect behavior, and same-region availability for a future head-to-head run.

I think that is exactly the right posture. The project does not delete the planned path or quietly rewrite history to pretend Phase 4 was always “just an Ada benchmark”. Instead, it distinguishes clearly between what was intended, what was possible at implementation time, and what remains future work.

That distinction is not a weakness. It is part of the credibility of the demo. Engineers can work with “measured on Ada, Blackwell pending.” They cannot do much with “trust me, the unavailable GPU would have been amazing”.

## Where this leaves the series

By the end of Phase 4, the project has a measured single-GPU cost baseline on Akamai Cloud, built on the RTX 4000 Ada node pool that was actually available in the cluster. The benchmark shows that concurrency can rise from 1 to 16 without a latency cliff, throughput can scale from roughly 21 tok/s to roughly 185 tok/s, and output-token cost can drop from roughly $12.77 to roughly $1.44 per million tokens in the measured prompt≈256-token case.

That is a good ending for the series because it ties the architecture back to economics. Reuse is not interesting only because it is technically elegant. It is interesting because wasted inference work eventually appears as lower throughput, worse utilization, and more dollars spent per useful token.

That closes the arc: Part 1 explains the mechanism, Part 2 proves cross-request reuse, and Part 3 turns the result into a cost model.

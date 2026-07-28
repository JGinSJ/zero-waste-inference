# Zero-Waste Inference, Part 1 — Building the KV cache from scratch

So the savings stop feeling like magic

*Part 1 of three, following the overview. This is Phase 1 in the [project repo](https://github.com/JGinSJ/zero-waste-inference); the repo runs its own phase numbering, which does not line up with the parts here.*

In the overview article, I argued that inference is often more wasteful than it is mysterious. Now it’s time to start at the very bottom of that problem and build the first piece ourselves: the key-value cache inside a decoder-only transformer [file:75][file:84].

This phase is intentionally small and intentionally unfancy. There is no Kubernetes cluster, no API gateway, no vLLM, no Fermyon, no Valkey, and no GPU deployment to distract us. The whole point is to make one very specific mechanism visible: once a model has computed the attention K and V tensors for prior tokens, it does not need to recompute them on every decode step [file:84][file:78].

If you’ve ever used a modern inference framework and thought, “Sure, I know KV caching helps,” but couldn’t quite point to where the saved work actually lives, this phase is for you. We’re going to build the thing from scratch, run it in both cached and non-cached modes, and make sure the outputs match before we trust the speedup [file:78][file:84].

## Why start here?

A lot of inference discussions begin at the serving layer, where the system is already buried under abstractions and flags and acronyms. That is useful later, but it is not a great place to learn what is actually being reused [file:84][file:75].

Phase 1 exists because I wanted a demo that makes the KV cache visible enough to inspect, test, and time directly in PyTorch, without leaning on HuggingFace, vLLM, or any other inference framework dependency [file:84][file:78]. The repo classifies this phase as a local demo rather than a deployed service, and that is deliberate: the purpose is not production throughput, but conceptual clarity you can carry into the rest of the stack [file:78][file:75].

The idea is simple. During autoregressive decoding, the attention keys and values for past positions do not change once they are computed, so they can be stored and reused rather than recomputed for every new token [file:78]. That’s the whole trick. The rest is implementation detail, debugging, and me trying not to pretend implementation detail and debugging are different things.

## What the demo is supposed to prove

This phase had a very narrow goal: build a minimal PyTorch transformer that makes the KV cache visible, measurable, and understandable [file:84]. Not “build an inference engine.” Not “optimize a production stack.” Just demonstrate exactly what is being reused and what is being recomputed on each forward pass so the later phases have something concrete to stand on [file:84][file:78].

The demo takes a short prompt, generates a continuation, records timing, and exposes cache growth as sequence length increases [file:84]. More importantly, it runs the same model two ways: once with a cache and once without one, then checks that both paths generate identical outputs so any speedup is tied to reuse rather than a correctness bug [file:78].

That last part matters more than it sounds. It is very easy to write “fast” code that is only fast because it is quietly wrong.

## Step 1 — Keep the model small enough to understand

The implementation is intentionally minimal: a decoder-only transformer with 4 layers, 4 heads, `d_model=256`, and `d_ff=1024`, plus a tiny character-level tokenizer with a 98-token vocabulary made from printable ASCII plus PAD, BOS, and EOS tokens [file:84]. This is not because small models are inherently noble. It is because if the model is too large or the stack is too abstract, the mechanism disappears behind the machinery [file:84][file:78].

The code lives under `phases/phase1-kv-cache/kv_cache/` and is split into four main modules: `attention.py`, `model.py`, `tokenizer.py`, and `generate.py` [file:78][file:84]. That split was not part of the original assumption, by the way. The plan initially leaned toward a simpler `kv_cache.py` file, but the implementation became much easier to reason about once the cache logic, model structure, tokenizer, and generation loop were separated into small files with one job each [file:78].

There is also a top-level `demo.py`, which runs the benchmark and the correctness check on the same randomly-seeded model and prompt, then writes the results to `results/phase1_timing.json` [file:78]. So yes, the “demo” and the “benchmark” ended up being the same program. I’m sure there is a more elegant way to say that, but the practical translation is: one script, less confusion [file:78].

## Step 2 — Make the KV cache explicit in attention

The heart of the phase is `MultiHeadAttention`, which accepts an optional `kv_cache` tuple containing the accumulated key and value tensors [file:78]. When there is no cache, the module computes fresh K and V tensors for the full input sequence, applies causal masking when needed, and returns both the attention output and a newly built cache [file:78].

When a cache is present, the behavior changes in the way we want. The layer computes K and V only for the new token, concatenates those tensors onto the cached K and V from prior positions, and attends over the combined sequence rather than recomputing history from scratch [file:78]. For single-token decode steps, no causal mask is needed because the query is already the latest position and is allowed to attend to everything that came before it [file:78].

This is the core of the savings. In the no-cache path, every new token drags the whole prior sequence back through attention again. In the cache path, the prior keys and values are treated as reusable state instead of freshly discovered ancient wisdom [file:78][file:84].

## Step 3 — Thread cache state through the full model

Attention alone is not enough. The model has to carry one cache per layer, update those caches as decoding proceeds, and make sure positional offsets continue from the already-cached sequence length instead of resetting back to zero [file:78].

That work lives in `model.py`, where each `TransformerBlock` returns both its transformed output and its updated `(K, V)` tuple, and `DecoderTransformer` accepts a list of per-layer caches for cached decode mode [file:78]. The positional offset is especially important here. If cached decode starts numbering positions from zero again instead of from the existing cache length, the model will be “efficient” in exactly the way a broken watch is “accurate twice a day” [file:78].

This is one of those details that gets hidden in big frameworks. When you build it yourself, you realize the cache is not a magical optimization flag. It is extra model state that must stay aligned with token position, layer structure, and decode order [file:78].

## Step 4 — Run generation in two distinct modes

The generation loop is where the difference becomes easy to explain. In cached mode, the model does one prefill pass over the full prompt, stores the KV states for each layer, and then runs one forward pass per new token using only the new token ID plus the accumulated cache list [file:84][file:78].

In no-cache mode, the model receives the entire growing sequence at every decode step with `kv_caches=None`, forcing it to recompute all keys and values for the whole prefix every single time [file:78]. Both paths return the generated token IDs, per-step timing, and final cache size in bytes, which means the demo can compare both speed and correctness in one place [file:78].

This is also why the script is useful as a teaching tool. “With cache” and “without cache” are not theoretical descriptions here. They are two concrete code paths you can time, inspect, and compare without a serving stack getting in the way [file:78][file:84].

## Step 5 — Refuse to trust a speedup until correctness passes

One of the design choices I liked most in this phase was the decision to test logit-level correctness, not just token-level equality [file:78]. The test suite includes an end-to-end token sequence check, but the stronger test compares the raw logits at every decode step using `torch.allclose(atol=1e-5)` so subtle cache bugs cannot hide behind matching argmax tokens [file:78].

There are 10 tests in total across cache correctness, KV shape growth, causal masking, and tokenizer behavior, and all 10 pass on the development machine used for the phase [file:78]. The tests verify things like cache length increasing by exactly one per decode step, earlier positions remaining unaffected by future tokens, and BOS/EOS/PAD token handling in the tokenizer [file:78].

That might sound like overkill for a toy demo. It is not. If this phase is supposed to justify the rest of a project about reuse, it has to prove that reuse did not quietly alter the model’s behavior [file:78][file:84].

## What the benchmark actually showed

The committed `results/phase1_timing.json` run was measured on the development machine using Python 3.11.15, PyTorch 2.2.2, and CPU execution on an Intel Mac [file:78]. The measured run used a 129-token prompt and generated 64 new tokens [file:78].

On that run, the cached path produced a prefill time of 10.6 ms and an average decode time of 1.89 ms per step, while the no-cache path averaged 6.98 ms per decode step [file:78]. The final cache size was about 1.5 MB, output correctness passed, and the measured decode speedup was 3.69× [file:78].

Those numbers are real measured results, not placeholders, but they are also not trying to be production throughput numbers [file:78]. The model is intentionally tiny, the hardware is just the dev machine, and the value of the benchmark is in making the compute scaling visible, not in pretending a laptop CPU run is now some industry-shaking performance statement [file:78].

## The friction points that were actually worth caring about

A few implementation details turned out to matter more than I expected. One was causal masking when mixing a cache of past positions with newly computed keys and values, which is why the mask logic is built position-by-position instead of relying on a fixed upper-triangular pattern [file:78]. Another was positional indexing during cached decode, which has to continue from the existing cache length rather than start over at zero [file:78].

There was also a tooling constraint on the dev machine: PyTorch 2.4+ was not available for the Intel Mac environment used for this phase, so `requirements.txt` was kept compatible with `torch>=2.1`, and the implementation sticks to stable pre-2.4 primitives [file:78]. That ended up being fine, because Phase 1 uses basic PyTorch operations only and does not depend on newer APIs [file:78].

And then there is the question I asked early on: should this run on the Akamai LKE cluster too? In the end, no Kubernetes manifest was written for Phase 1, because the model runs in about 2 seconds on CPU, has no operational dependency on GPU hardware, and is meant to be a conceptual building block rather than a deployed service [file:78][file:75]. Sometimes the correct MLOps decision is not to add more MLOps.

## Where this leaves us

By the end of Phase 1, we have a runnable PyTorch demo that makes KV caching visible and measurable, a paired cached versus non-cached generation path, a correctness test suite, and a real timing result showing a 3.69× decode speedup on the dev machine for the measured run [file:78][file:84]. That is enough to establish the first and most local form of reuse in the project: don’t recompute attention history that the model already knows [file:84][file:78].

In Part 2, I’m going to move up the stack from within-request reuse to across-request reuse. That means Fermyon Wasm Functions at the front door, Valkey as the cache layer, and a vLLM backend on Akamai LKE for the misses that still need GPU work [file:85][file:75]. The mechanism gets more distributed and the plumbing gets more interesting, but the logic stays the same: stop throwing away useful compute just because the request boundary changed [file:85][file:78].

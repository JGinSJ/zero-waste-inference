# Public article drafts

The public-facing series for Medium. **Series part numbers and repo phase
numbers are deliberately different** — the repo has four phases; the series has
an overview plus three parts.

| Series | File | Repo phase | Title |
|---|---|---|---|
| Overview | *(not in repo — unpublished)* | — | Zero-Waste Inference on Akamai Cloud |
| Part 1 | [`zero_waste_inference_part1_article.md`](zero_waste_inference_part1_article.md) | Phase 1 | Building the KV cache from scratch |
| Part 2 | [`zero_waste_inference_part2_article.md`](zero_waste_inference_part2_article.md) | Phase 2 | Caching at the front door |
| Part 3 | [`zero_waste_inference_part3_article.md`](zero_waste_inference_part3_article.md) | Phase 4 | What the GPU actually costs |

## Why the numbering differs

**Phase 3 (Qwen image serving) is not in the series.** It is built, deployed,
and documented — see [`docs/phase3-qwen-image-build-log.md`](../phase3-qwen-image-build-log.md) —
but it tells the same reuse story with a different payload, and a gap in a
published series ("…where's part 3?") costs more reader trust than it buys.
Rather than publish four parts with an unexplained jump, the series runs
1 → 2 → 3 and Part 3 notes the omission in a single line.

The repo keeps phases 1–4 throughout. Build logs, scope docs, manifests, and
CLAUDE.md all depend on that numbering, and it is accurate. Each article carries
a line near the top mapping its part number to its repo phase; where an article
discusses the repo's work it says "Phase N", and where it refers to the reading
sequence it says "Part N".

## Citation markers

The drafts carry `[file:NN]` markers from the source-gathering pass. They are
not Markdown links and will render literally — **strip them before publishing.**
Rough mapping:

| Marker | Source |
|---|---|
| `[file:75]` | root `README.md` |
| `[file:77]` | `docs/architecture.md` |
| `[file:78]` | `docs/phase1-kv-cache-build-log.md` |
| `[file:79]` | `docs/phase2-fermyon-build-log.md` |
| `[file:82]` | `docs/phase4-build-log.md` |
| `[file:83]` | `docs/phases/phase4-benchmarks.md` |
| `[file:84]` | `docs/phases/phase1-kv-cache.md` |
| `[file:85]` | `docs/phases/phase2-fermyon-valkey.md` |
| `[file:86]` | `docs/phases/phase3-qwen-image.md` |

## Accuracy notes

Every measured figure in these drafts was checked against the committed results
(`phase1_timing.json`, `phase2_cache_benchmark.json`,
`results/phase4_raw_benchmark.csv`) and against a live `pytest` run. Two claims
to keep straight, because both were wrong in earlier drafts:

- The Fermyon front door is an **exact-match response cache**, not a prefix
  cache. Prefix reuse happens one layer down, in LMCache. vLLM runs
  `--no-enable-prefix-caching` because the two do not stack.
- The **25.9%** hit rate belongs to LMCache's KV cache, measured direct-to-vLLM
  before the front door was wired up. It is not the response cache's hit rate
  and must not be compared against the 1.3% break-even.

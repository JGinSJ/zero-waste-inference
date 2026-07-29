# Booth demo — the two-lane race

A full-screen display that races two identical requests against each other: one
through the Fermyon front door, one straight at the GPU. The point is the dead
time — the cached lane finishes in ~218 ms while the direct lane crawls for
three full seconds, and the audience watches it happen.

```
demo/
├── booth-race.html   ← the display (self-contained; no external assets)
├── relay.py          ← local relay: proxies both lanes, serves the page
└── README.md         ← this file
```

---

## Quick start

**Prerequisites:** `git` and `python3`. That is the whole list — the relay is
Python standard library only, so there is no `pip install` and no build step.
Any Python 3.7 or newer works; macOS and Linux already ship one.

```bash
git clone https://github.com/JGinSJ/zero-waste-inference.git
cd zero-waste-inference
python3 demo/relay.py --replay-only
```

Open **http://localhost:8099** and press `F` for fullscreen.

| Key | Does |
|---|---|
| `space` | run a race |
| `F` | fullscreen |
| `R` | receipts — raw JSON, response headers, timings |
| `Esc` | close receipts |

Tap any question button to run that one. Leave it alone for 20 seconds and it
demos itself, so the screen is never static.

**What you are looking at:** two identical requests race each other — one
through the cache, one straight to the GPU. The first ask is a cold miss and
both take about three seconds. Ask the same thing again and the cached lane
returns in ~218 ms while the GPU lane still grinds.

**The demo replays measured data. It does not call a live cluster.** Every
latency on screen is a real sample from a benchmark run on Akamai LKE on
2026-04-16 — nothing is simulated, and nothing is generated at the booth. The
badge in the top right says `REPLAY` with the source file and date, permanently.
That is the design, not a fallback: an expo hall is the last place to bet a
demo on network connectivity, and replayed measurements make exactly the same
point as a live call while being immune to the venue's wifi.

So the demo needs **no network, no cluster, no GPU, and no credentials.** It
runs identically on a plane.

### Notes for anyone running this cold

- **Port 8099 already taken?** `--port 8123` moves it. The page follows
  automatically; it derives the relay endpoint from its own origin.
- **The relay binds `0.0.0.0`, not localhost.** That is deliberate — you can
  point a second screen or a tablet at the booth laptop's IP. It also means
  anyone on the same network can load the page. Fine at a stand, worth knowing
  on café wifi. Change the bind address in `relay.py` if you would rather it
  stayed private.
- **Browser:** Safari 16.2+ or Chrome 111+ (the page uses `color-mix()` and
  `crypto.randomUUID()`). Serve it through the relay rather than opening the
  `.html` off disk — `randomUUID` needs a secure context, and `http://localhost`
  counts as one.

---

## Why there is a relay at all

For the booth, the relay does one job: serve the page over `http://` instead of
`file://`. That matters more than it sounds — `crypto.randomUUID()` needs a
secure context, and `http://localhost` qualifies while `file://` does not.

The relay also contains a live proxy path, which is not used at the booth and is
documented in `relay.py` rather than here. Short version: a browser cannot read
`X-Cache: HIT|MISS` off the cluster directly, because neither the Fermyon proxy
nor vLLM sends CORS headers — so when the data is regenerated, the relay
proxies each lane and times it server-side. That path is how
`phase2_cache_benchmark.json` gets refreshed, not how the booth runs.

---

## How the demo runs — two acts

Each race is deliberately two acts, because a one-act version would imply the
cache is magic:

**Act 1 — a new question.** A uuid4 nonce is embedded in the message, so this is
a guaranteed cold miss (the same cold-cache method as `benchmark/bench_cache.py`,
and it avoids needing `FLUSHDB` on Valkey). Both lanes take ~3 s. The badge reads
`MISS`. Caption: *"Both paths paid full price. That is the honest cost of a new
question."*

**Act 2 — the same question again.** Identical payload, nonce and all. The
Fermyon lane hits Valkey and returns in ~218 ms; the direct lane still pays the
GPU. Badge reads `HIT`, the GPU pill reads *"GPU never woke up"*, and the hero
number lands at roughly 13–14×.

The nonce is disclosed in the receipts panel — it is cache-busting, not sleight
of hand.

---

## Operating it

| Key | Action |
|---|---|
| `space` | run the next preset question |
| `R` | receipts panel (raw JSON, both acts, headers, nonce) |
| `F` | fullscreen |
| `Esc` | close receipts |

(`L` toggles live/replay during development. When the relay runs
`--replay-only` the key is refused, so the badge can never read `LIVE` above
replayed numbers.)

Tapping any preset button runs that question. After 20 s idle the attract loop
runs a race on its own, so the screen is never static while you are talking to
someone.

Counters persist in `localStorage` across refreshes and survive a laptop
sleep — reset them from the receipts panel at the start of each show day.

---

## Show-day checklist

1. **Nothing to warm up, nothing to connect.** No cluster, no port-forwards, no
   credentials, no wifi. Start the relay, open the page, done. If the venue
   network is down, or you never joined it, the demo does not notice.
2. **Reset the counters.** Receipts panel → *Reset booth counters*. They persist
   in `localStorage`, so yesterday's totals survive overnight unless you clear
   them.
3. **Fullscreen (`F`) and check from ten feet.** Everything is sized in `vh`/`vw`
   units; the type should be readable from across the aisle.
4. **Run one race before doors open** to confirm the display resolution looks
   right — the layout adapts, but you want to see it on the actual screen.
5. Have the receipts panel ready. The skeptical engineer is your best lead, and
   the raw JSON with `X-Cache: HIT` is what wins them.
6. **Say it replays measured data**, ideally before anyone asks. It costs a
   sentence, it is on the screen anyway, and it is a better story than pretending
   a booth laptop is talking to Chicago.

---

## Honesty notes (these are load-bearing)

- **The booth demo replays measured data, and the screen says so.** The `REPLAY`
  badge carries the source file and the measurement date at all times. Nothing
  here claims to be a live inference call, and the `L` key cannot be used to
  make it look like one.
- **The cold-start outlier is excluded from replay.** The 8,173 ms first request
  in `pass1_cold` is a documented vLLM cold start, not a representative miss;
  replaying it in a loop would misstate typical MISS latency. The other nine
  pass-1 samples are used as measured.
- **"GPU time avoided" is a concurrency-1 figure.** A hit avoids ~3.0
  GPU-seconds *at c=1*. At c=16 the same request costs roughly 8.8× less GPU
  time (Phase 4), so the counter is labelled `measured at concurrency 1` on
  screen. Say so if anyone asks — someone will.
- **The `$805 per 1M hits` figure is a projection**, marked as such on the tile.
  It is 3.02 s × 1M = 839 GPU-hours × $0.96/hr, nothing more.
- **This layer is an exact-match response cache, not a prefix cache.** The key
  is `SHA-256(model + "\x00" + messages)` over the whole message array. Two
  requests sharing a long prefix but diverging at the end do *not* hit. Prefix
  reuse happens one layer down, in LMCache. Act 2 hits because it is a byte-for-byte
  repeat.

---

## Colour

Two series, dark surface, from the data-viz reference palette:
slot 1 blue `#3987e5` (cached path) and slot 2 orange `#d95926` (GPU path).
Validated against the `#1a1a19` surface — worst-pair CVD ΔE 26.8 (protan),
normal-vision ΔE 31.8, both ≥ 3:1 contrast. Identity never rests on colour
alone: every lane carries a swatch, a name, and a text badge.

//! Fermyon Wasm caching proxy — Phase 2
//!
//! # Overview
//!
//! Transparent HTTP proxy that sits in front of vLLM.
//!
//! ```text
//! POST /v1/chat/completions
//!        │
//!        ▼
//!   parse + validate
//!        │
//!        ▼
//!   "fermyon:v1:" + hex(SHA-256(model + "\x00" + messages_json))
//!        │
//!     GET key from Valkey
//!    /               \
//!  HIT               MISS
//!   │                  │
//!   │              POST → vLLM /v1/chat/completions
//!   │                  │
//!   │              SET key in Valkey  ← best-effort; never fails request
//!   │              EXPIRE key TTL s   ← best-effort
//!   │                  │
//!   └──────────────────┘
//!        │
//!        ▼
//!   raw vLLM JSON body + X-Cache: HIT/MISS header
//! ```
//!
//! # Cache key
//!
//! ```text
//! key = "fermyon:v1:" + hex(SHA-256(model + "\x00" + messages_json))
//! ```
//!
//! Only `model` and `messages` (role + content) contribute to the key.
//! Sampling parameters (`temperature`, `top_p`, `max_tokens`, etc.) and
//! `stream` are intentionally excluded so that requests with identical
//! prompts but different sampling settings share a cache entry.
//!
//! The null-byte separator between `model` and `messages_json` prevents
//! hash collisions when a model name ends with characters that appear in
//! valid JSON.
//!
//! Message objects are normalised to `{role, content}` only before
//! serialisation, giving a consistent cache key regardless of whether the
//! client sends fields in `{role, content}` or `{content, role}` order.
//!
//! # Streaming limitation
//!
//! The Fermyon Wasm runtime does not support streaming HTTP responses.
//! If `stream: true` is present in the request it is **stripped** before the
//! upstream vLLM call — and the event is logged to stderr, not swallowed;
//! the complete response is returned as a single JSON body.  Clients relying
//! on SSE / chunked-transfer will not receive tokens incrementally.  This is
//! a known Phase 2 limitation.
//!
//! # TTL
//!
//! Entries are written with a TTL of `cache_ttl` seconds (default 3600).
//! The TTL is **not reset on cache hits** — it is fixed from first write.
//!
//! # Key namespace
//!
//! All keys written by this handler carry the `"fermyon:v1:"` prefix.
//! LMCache writes chunk-hash keys without this prefix; the two sets
//! do not collide in Valkey.

use anyhow::{anyhow, Context};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use spin_sdk::{
    http::{IntoResponse, Method, Request, Response},
    http_component,
    redis::Connection as RedisConn,
    variables,
};

/// Namespace prefix applied to every Valkey key written by this handler.
/// Ensures no collision with LMCache chunk-hash keys or other Valkey tenants.
const CACHE_KEY_PREFIX: &str = "fermyon:v1:";

/// Default TTL in seconds if the `cache_ttl` variable is absent or unparseable.
const DEFAULT_CACHE_TTL: i64 = 3600;

// ---------------------------------------------------------------------------
// Normalised message type used solely for cache key derivation
// ---------------------------------------------------------------------------

/// A single chat message, normalised to role + content only.
///
/// Used exclusively to compute the cache key.  Extra fields present in the
/// incoming request (`name`, `tool_call_id`, etc.) are dropped on
/// deserialisation, ensuring that two requests with the same logical
/// messages but different extra fields produce the same cache key.
#[derive(Deserialize, Serialize)]
struct Message {
    role: String,
    content: String,
}

// ---------------------------------------------------------------------------
// Cache key
// ---------------------------------------------------------------------------

/// Compute the Valkey cache key for a chat completion request.
///
/// # Algorithm
///
/// 1. Serialise `messages` (already normalised to `[{role, content}]`) to
///    a compact JSON string.
/// 2. Feed `model`, a null-byte separator, and the messages JSON into SHA-256.
/// 3. Prepend `"fermyon:v1:"` to the lowercase hex digest.
///
/// # Key properties
///
/// - Deterministic: same `model` + `messages` always produces the same key.
/// - Field-order-independent: messages are normalised before serialisation.
/// - Excludes sampling params: `stream`, `temperature`, `top_p`, `max_tokens`.
/// - Null-byte separator prevents prefix collisions.
pub fn make_cache_key(model: &str, messages: &[Message]) -> String {
    let messages_json = serde_json::to_string(messages).unwrap_or_default();
    let mut hasher = Sha256::new();
    hasher.update(model.as_bytes());
    hasher.update(b"\x00");
    hasher.update(messages_json.as_bytes());
    let digest = hasher.finalize();
    format!("{CACHE_KEY_PREFIX}{digest:x}")
}

// ---------------------------------------------------------------------------
// Response helpers
// ---------------------------------------------------------------------------

fn error_response(status: u16, message: &str) -> Response {
    Response::builder()
        .status(status)
        .header("content-type", "text/plain")
        .body(message.to_string())
        .build()
}

/// Return `body` as-is with Content-Type: application/json and an X-Cache
/// header.  Transparent passthrough — no response wrapping.
fn passthrough_response(body: Vec<u8>, x_cache: &str) -> Response {
    Response::builder()
        .status(200)
        .header("content-type", "application/json")
        .header("x-cache", x_cache)
        .body(body)
        .build()
}

// ---------------------------------------------------------------------------
// HTTP component entry point
// ---------------------------------------------------------------------------

#[http_component]
async fn handle(req: Request) -> anyhow::Result<impl IntoResponse> {
    // Spin's route-based dispatch guarantees this handler only receives
    // requests matched by the /v1/chat/completions trigger route in spin.toml.
    // No path inspection is needed here.

    // -----------------------------------------------------------------------
    // Only POST is accepted
    // -----------------------------------------------------------------------
    if *req.method() != Method::Post {
        return Ok(error_response(405, "Method Not Allowed — use POST"));
    }

    // -----------------------------------------------------------------------
    // Read Spin variables
    // -----------------------------------------------------------------------
    let valkey_addr = variables::get("valkey_address")
        .context("Spin variable 'valkey_address' is required but not set")?;

    let vllm_url = variables::get("vllm_url")
        .context("Spin variable 'vllm_url' is required but not set")?;

    let cache_ttl: i64 = variables::get("cache_ttl")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_CACHE_TTL);

    // -----------------------------------------------------------------------
    // Parse request body
    //
    // Parse as a generic JSON Value first so that unknown fields are
    // preserved when forwarding to vLLM.  Then extract and normalise
    // `model` and `messages` for cache key derivation.
    // -----------------------------------------------------------------------
    let body_bytes = req.body().to_vec();

    let mut body_value: serde_json::Value = serde_json::from_slice(&body_bytes)
        .map_err(|e| anyhow!("invalid JSON body: {e}"))?;

    let model = body_value["model"]
        .as_str()
        .ok_or_else(|| anyhow!("'model' field missing or not a string"))?
        .to_string();

    let messages_value = body_value
        .get("messages")
        .ok_or_else(|| anyhow!("'messages' field missing"))?
        .clone();

    // Normalise messages to [{role, content}] — drops extra fields and
    // ensures consistent field order for cache key serialisation.
    let messages: Vec<Message> = serde_json::from_value(messages_value)
        .map_err(|e| anyhow!("'messages' field invalid: {e}"))?;

    if messages.is_empty() {
        return Ok(error_response(400, "'messages' must be a non-empty array"));
    }

    // -----------------------------------------------------------------------
    // Strip stream: true
    //
    // LIMITATION: The Fermyon Wasm runtime does not support streaming HTTP
    // responses.  If stream:true is present it is stripped before forwarding.
    // The upstream vLLM call is made as a standard (non-streaming) request
    // and the complete JSON response is returned to the client.
    // Clients expecting SSE / chunked-transfer will not receive incremental
    // tokens.  This is a known Phase 2 limitation.
    // -----------------------------------------------------------------------
    if let Some(obj) = body_value.as_object_mut() {
        let was_streaming = obj
            .remove("stream")
            .as_ref()
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        if was_streaming {
            eprintln!(
                "fermyon-prefix-cache: stream:true stripped — streaming not supported; \
                 returning complete non-streaming response"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Cache lookup
    // -----------------------------------------------------------------------
    let cache_key = make_cache_key(&model, &messages);

    // Connect to Valkey.
    // The redis:// URL scheme is what the Spin Redis SDK expects.
    // The actual backend is Valkey (open-source Redis-compatible fork) —
    // not Redis itself.  Valkey speaks the RESP protocol natively.
    let conn = RedisConn::open(&valkey_addr)
        .context("failed to open Valkey connection")?;

    match conn.get(&cache_key) {
        Ok(Some(cached_bytes)) => {
            // Cache HIT.
            // Return the stored response body exactly as written on the
            // original miss.  TTL is NOT reset on hits — it was fixed at
            // write time and counts down independently.
            return Ok(passthrough_response(cached_bytes, "HIT"));
        }
        Ok(None) => {
            // Cache miss — continue to vLLM below.
        }
        Err(e) => {
            // Valkey read error — treat as a cache miss so the request is
            // served rather than failed.  Log for observability.
            eprintln!(
                "fermyon-prefix-cache: Valkey GET error (key={cache_key}): {e}; \
                 treating as miss"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Cache MISS — forward to vLLM
    // -----------------------------------------------------------------------
    let forward_body = serde_json::to_vec(&body_value)
        .context("failed to re-serialise request body for forwarding")?;

    let vllm_req = Request::builder()
        .method(Method::Post)
        .uri(format!("{vllm_url}/v1/chat/completions"))
        .header("content-type", "application/json")
        .body(forward_body)
        .build();

    let vllm_resp: Response = spin_sdk::http::send(vllm_req)
        .await
        .context("upstream vLLM request failed")?;

    let vllm_status = *vllm_resp.status();

    if vllm_status != 200 {
        return Ok(error_response(
            vllm_status,
            &format!("upstream vLLM returned HTTP {vllm_status}"),
        ));
    }

    let vllm_body = vllm_resp.body().to_vec();

    // Store in Valkey — both calls are best-effort.
    // A write or expire failure must not cause the client request to fail;
    // the next identical request will simply be another cache miss.
    let _ = conn.set(&cache_key, &vllm_body);
    let _ = conn.execute("EXPIRE", &[
        spin_sdk::redis::RedisParameter::Binary(cache_key.as_bytes().to_vec()),
        spin_sdk::redis::RedisParameter::Int64(cache_ttl),
    ]);

    Ok(passthrough_response(vllm_body, "MISS"))
}

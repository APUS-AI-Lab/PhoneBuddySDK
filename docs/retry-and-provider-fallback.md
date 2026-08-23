# Retry and Provider Fallback

PhoneBuddy retries transient LLM failures locally, and — when the host
supplies a fallback chain — fails over to the next provider in seconds
rather than burning a full retry budget on a dead endpoint.

The retry/failover state machine lives in the engine
(`crates/phone-buddy/src/llm/client.rs`, `retry.rs`, `failover.rs`).
The host app chooses which providers to put on the chain, in which
order, and how to surface `Retrying` / `ProviderSwitched` events.

## 1. Single-provider retry (default)

When `fallback_providers` is empty, behaviour matches the grok sampler
policy. `EngineConfig.max_retries` (default **5**) is the attempt
budget.

| Error class | Examples | Action |
|---|---|---|
| `Retry` | 5xx except 525/526, connection errors, timeouts, empty responses | Exponential backoff 2s → 4s → 8s → 16s → 30s (±20% jitter) |
| `RateLimited` | HTTP 429 | Honor `Retry-After` (parse cap 120s). Independent budget of 2 retries |
| `Fatal` | 400/401/403/404/408/422, 525/526 | Fail immediately |
| Veto | Context-window overflow, `x-should-retry: false` | Fail immediately, never retry |

Retries only happen **before any delta has been streamed** to the
observer. A mid-stream failure is surfaced as-is so the UI never sees
duplicated tokens.

The engine emits a `Retrying` event before each wait:

```json
{"Retrying":{"provider":"api.example.com/grok-4.6","attempt":2,"max_attempts":5,"wait_ms":2100,"reason":"status=503"}}
```

`provider` is a desensitized `host/model` fingerprint. It never
includes an API key.

## 2. Chain mode (fallback providers)

`EngineConfig` fields:

```json
{
  "base_url": "https://primary.example/v1",
  "api_key": "...",
  "model": "grok-4.6",
  "fallback_providers": [
    {
      "base_url": "https://api.openai.com/v1",
      "api_key": "...",
      "model": "gpt-5.6",
      "api_backend": "chat_completions",
      "client_profile": "default",
      "enable_web_search": false
    }
  ],
  "failover_max_attempts": 3,
  "provider_cooldown_secs": 120
}
```

When the chain is non-empty:

- Each provider is limited to `failover_max_attempts` (default **3** =
  initial + 2 retries, ~2s + ~4s backoff ≈ 6s).
- `max_retries` is ignored for failover decisions. Empty-response and
  server doom-loop resampling keep their own independent budgets and
  do **not** trigger a provider switch.
- After the per-provider budget is spent, the engine marks that
  provider degraded and moves to the next endpoint.

### Failover triggers

| Error class | In-provider | Switch? |
|---|---|---|
| `Retry` (503/5xx, connect, timeout) | Backoff 2s → 4s, 3 tries | Yes, after budget |
| `RateLimited` (429) | Wait in place only if wait ≤ 10s (budget 2) | Yes if `Retry-After` > 10s or budget spent. Cooldown = `max(Retry-After, computed)` |
| `Fatal` non-veto (400/401/403/404/525/526) | No retry | **Immediately** |
| Veto (context overflow, `x-should-retry: false`) | No retry | **No** — terminate the request |
| Doom-loop / empty response | Existing local budget | No |

A `ProviderSwitched` event is emitted on every switch:

```json
{"ProviderSwitched":{"from":"primary.example/grok-4.6","to":"api.openai.com/gpt-5.6","reason":"LLM request failed: status=503 …","cooldown_ms":120000}}
```

If every provider in the chain is exhausted, the last error is
returned with a `[tried: host-a/m, host-b/m]` suffix.

Worst-case wait for N providers is about `N × 6s` (N=3 ≈ 18s), versus
~60s on a single provider with `max_retries = 5`.

## 3. Cooldown and stickiness

Each engine instance holds a per-provider health table:

- First trip → sit out `provider_cooldown_secs` (default 120s).
- Consecutive trips double the cooldown: 120 → 240 → 480 → 600s (cap).
- A successful probe resets the table.

**Selection** (every LLM request, including each hop of a tool loop):
take the first provider in chain order that is not cooling. If all are
cooling, take the one whose cooldown expires soonest, so a request
always has a provider.

While the primary is cooling, subsequent turns go straight to the
backup with **no extra wait**. When the cooldown expires the primary
is passively re-probed on the next request — there is no background
health check.

The table lives in engine memory. Recreating the engine (config
change, process restart) clears it. Hosts that want stickiness across
recreates can persist `ProviderSwitched.cooldown_ms` and reorder the
chain on the next `createEngine`.

## 4. Cross-model history

The engine never uses `previous_response_id` / server-stored sessions
(same as grok-build: that field is always `None`). Every request
re-encodes the full in-memory `ChatMessage` history for the target
backend. Continuity comes from replaying the conversation items,
including Responses reasoning siblings (`rs_*` id + `encrypted_content`).

Assistant turns are tagged with a compatibility key `{provider_group}/{model}`.
`provider_group` defaults to the client profile name (`grok_build`,
`codex`, `claude_code`, `default`). Hosts may set it explicitly.

| Artifact | Same group + same model | Group or model changed |
|---|---|---|
| User / assistant text, tool calls + tool output | Kept | **Kept** |
| `reasoning_items` (`rs_*` id + encrypted Responses reasoning) | **Kept** (full grok-build replay) | **Stripped** |
| `encrypted_reasoning` (Anthropic thinking signature) | **Kept** | **Stripped** |
| Plain `reasoning_content` | **Kept** | **Stripped** |

So Hermes → another `grok_build` gateway on `grok-4.6` keeps encrypted
thinking. Switching to a Claude-profile endpoint, or staying in
`grok_build` but changing the model id, drops it. Text and tool
results always survive so an in-flight turn can continue.

## 5. CLI

```bash
PHONEBUDDY_API_KEY=... PHONEBUDDY_FALLBACK_API_KEY=... \
  cargo run -p phone-buddy-cli -- chat \
    --fallback-url https://api.openai.com/v1 \
    --fallback-model gpt-4o \
    "summarize data/sales.csv"
```

`--fallback-url` may be repeated. `--fallback-model` / `--fallback-key`
apply to the most recently added URL. If they are omitted, the CLI
falls back to `PHONEBUDDY_FALLBACK_MODEL` / `PHONEBUDDY_FALLBACK_API_KEY`,
then to the primary `PHONEBUDDY_MODEL` / `PHONEBUDDY_API_KEY`.

The printer logs `Retrying` and `ProviderSwitched` events on stderr.

## 6. Invariants

1. No retry or failover after the first streamed delta.
2. Veto errors never switch providers.
3. Fallback chain empty ⇒ identical to historical single-provider
   behaviour, including the large `max_retries` budget.
4. Provider identifiers in events never carry secrets.
5. Fallback endpoints inherit engine-level `stream_idle_timeout_secs`
   and `http_dump` (no per-endpoint override in this revision).

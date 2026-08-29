# Retry and Provider Fallback

PhoneBuddy retries transient LLM failures locally and fails over across the
providers in a named pool. Routing, health scoring, cooldown, and selection
are owned by the SDK (`crates/phone-buddy/src/llm/router/`, `client.rs`,
`retry.rs`). The host app supplies a declarative snapshot of servers, pool
membership, base scores, and grouping; it does not run a second router.

`EngineConfig` primary + `fallback_providers` still works. The engine
synthesizes a `main` pool (and a `subagent` copy of it) through a private
[`PhoneBuddyRuntime`] so existing callers keep working. There is **no**
implicit `session_title` pool; a missing utility pool returns
`RouteNotConfigured`.

## 1. Single-provider retry (default)

When the bound pool has one member (legacy: `fallback_providers` is empty),
behaviour matches the grok sampler policy. `EngineConfig.max_retries`
(default **5**) is the attempt budget.

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
{"Retrying":{"provider":"api.example.com/grok-4.6","attempt":2,"max_attempts":5,"wait_ms":2100,"reason":"status=503","provider_id":"legacy-primary","pool_id":"main","workload":"main"}}
```

`provider` is a desensitized `host/model` label kept for existing UI.
`provider_id` is the stable, secret-free join key. `workload` is `main`,
`subagent`, or `one_shot`, so a `provider_id` listed in several pools can
still be attributed to the work that tripped it. No field includes an API
key.

## 2. Named pools (and the legacy chain)

Preferred configuration is `LlmRoutingConfig`: named `ProviderTarget`s,
named pools (`main`, `subagent`, `session_title`, …), and a
`RouterHealthConfig`. Create a long-lived `PhoneBuddyRuntime` and bind
engines with `runtime.create_engine(agent_config, "main")`.

The main-agent tool loop uses the engine's pool-bound client (typically
`main`). `TaskManager` receives a **second** `LlmClient` bound to
`subagent`: same router (shared health), independent provider slots, and
a fresh `begin_turn()` per spawned subagent. A direct routing config that
omits `subagent` returns `RouteNotConfigured` when creating an HTTP
engine. Only the legacy `EngineConfig` adapter copies `main` into
`subagent`. Utility pools such as `session_title` stay unconfigured until
the host declares them.

Legacy `EngineConfig` fields still synthesize a chain:

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

The adapter produces:

- provider ids `legacy-primary`, `legacy-fallback-0`, …
- a `main` pool in declared order, routing group `default`, base score 10,
  `when_exhausted = probe_earliest`
- a `subagent` copy of `main`
- `reasoning_compatibility_key` = `{provider_group}/{model}` so encrypted
  reasoning behaviour does not regress

When the pool has more than one member:

- Each provider is limited to `failover_max_attempts` (default **3** =
  initial + 2 retries, ~2s + ~4s backoff ≈ 6s).
- `max_retries` is ignored for failover decisions. Empty-response and
  server doom-loop resampling keep their own independent budgets and
  do **not** trigger a provider switch.
- After the per-provider budget is spent, the engine records **one** trip
  for that logical visit and moves to the next endpoint.

### Failover triggers

| Error class | In-provider | Switch? |
|---|---|---|
| `Retry` (503/5xx, connect, timeout) | Backoff 2s → 4s, 3 tries | Yes, after budget |
| `RateLimited` (429) | Wait in place only if wait ≤ 10s (budget 2) | Yes if `Retry-After` > 10s or budget spent. Cooldown = `max(Retry-After, computed)` |
| `Fatal` non-veto (400/401/403/404/525/526) | No retry | **Immediately** |
| Veto (context overflow, `x-should-retry: false`) | No retry | **No** — terminate the request |
| Doom-loop / empty response | Existing local budget | No (empty responses in a multi-member pool do switch after the per-provider budget) |

A `ProviderSwitched` event is emitted on every switch. Stable ids are the
join key; `from` / `to` remain as sanitized labels during deprecation:

```json
{"ProviderSwitched":{"from":"primary.example/grok-4.6","to":"api.openai.com/gpt-5.6","reason":"LLM request failed: status=503 …","cooldown_ms":120000,"from_provider_id":"legacy-primary","to_provider_id":"legacy-fallback-0","pool_id":"main","workload":"main","failure_class":"retryable_http"}}
```

If every provider in the visit plan is exhausted, the SDK returns
`ProviderAttemptsExhausted` with the tried `provider_id`s. If a `fail_fast`
pool has no eligible member, it returns `PoolExhausted` with a retry-after
hint instead of probing a cooling endpoint.

Worst-case wait for N providers is about `N × 6s` (N=3 ≈ 18s), versus
~60s on a single provider with `max_retries = 5`.

## 3. Health, scoring, and persistence

Health is keyed by stable `provider_id` and **shared across pools**. Two
workloads that should fail independently must use two ids even if their
URLs match.

```text
effective_score = base_score - failures_within(penalty_window)
```

Default `penalty_window` is one hour. A successful operation clears
consecutive-trip cooldown escalation but does **not** erase historical
failures; they age out of the window. Retry attempts inside one provider
visit count as **one** trip (idempotent if recorded twice).

Cooldown progression is configurable (`RouterHealthConfig`); the default
shape is 120 → 240 → 480 → 600 seconds. A long `Retry-After` may trip
immediately with a longer cooldown.

### Selection (every LLM operation)

1. Drop disabled members and prune expired failure timestamps.
2. Prefer members with `effective_score > 0` that are not cooling.
3. Rank each `routing_group` by the maximum effective score among eligible
   members.
4. Break group ties by the lowest declared member `order`.
5. Within a group, order by effective score descending, then `order`
   ascending.
6. Visit that sequence, applying the pool retry budget before failover.

If nothing is normally eligible:

- `probe_earliest` (default for `main`): enabled provider whose cooldown
  expires first; tie-break score then `order`.
- `fail_fast`: `PoolExhausted` with a retry-after hint.

While a preferred member is cooling, later operations go straight to the
next eligible member. When the cooldown expires the member is passively
re-probed — there is no background health check.

### Persistence

Non-secret health is stored at:

```text
<root_dir>/.phonebuddy/router/health-v1.json
```

Writes use a temp file and atomic rename, serialized behind one runtime
lock. Corrupt or unsupported snapshots are discarded (fail open). Records
for provider ids absent from the current config are dropped after a
bounded retention period. The file never contains URLs, credentials,
request bodies, prompts, response ids, encrypted reasoning, or
`x-codex-turn-state`.

`PhoneBuddyRuntime` is longer-lived than an engine. Recreating an engine
against the same runtime (or reloading the health file after process
restart) keeps cooldown and scores. A config update is reconciled by
stable `provider_id`.

Wall-clock timestamps are used so the snapshot can be serialized;
`std::time::Instant` is not.

## 4. Cross-model history

Normal user turns do not carry `previous_response_id` across turns. Every
request can re-encode the full local conversation-item history for the target
backend, and that local history remains the source of truth. The Responses
SSE idle-timeout recovery path may reuse a captured response ID once on the
exact same provider to continue the same logical LLM operation; provider
failover clears it.

Likewise, the `x-codex-turn-state` routing header is scoped to one logical
main-agent or subagent turn and one exact transport. It may be reused by
retries and tool-loop hops inside that turn, but never crosses user turns,
providers, or concurrently running subagents.

Continuity otherwise comes from replaying the conversation items, including
Responses reasoning siblings (`rs_*` id + `encrypted_content`).

Assistant turns are tagged with a `reasoning_compatibility_key`. A missing
key defaults to the unique `provider_id`. The legacy adapter preserves
`{provider_group}/{model}` (group defaults to the client profile name:
`grok_build`, `codex`, `claude_code`, `default`). `routing_group` is a
separate field used only for pool ranking; it is not cryptographic
compatibility.

| Artifact | Same compatibility key | Key changed |
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
2. Veto errors never switch providers and never poison health.
3. Fallback chain empty ⇒ identical to historical single-provider
   behaviour, including the large `max_retries` budget.
4. Provider identifiers in events never carry secrets. Stable
   `provider_id` is the join key; sanitized `host/model` is a label.
5. A provider trip is reported once per logical visit, not once per
   low-level retry.
6. Fallback endpoints inherit engine-level `stream_idle_timeout_secs`
   and `http_dump` (no per-endpoint override in this revision).
7. Health is shared by `provider_id` across pools and engines that
   borrow the same runtime.
8. Subagents never inherit the main pool by falling back; missing
   `subagent` is `RouteNotConfigured`. Free-form task `model` overrides
   are not accepted — the selected pool member supplies the model.
9. Every routed operation reports its `workload` (`main`, `subagent`,
   `one_shot`) alongside `pool_id` and `operation_id`. One-shot calls have
   no host event stream, so their routing diagnostics go to `tracing`.

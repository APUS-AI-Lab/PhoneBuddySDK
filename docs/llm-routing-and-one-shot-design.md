# LLM Routing Pools and One-Shot Generation

Status: implementation proposal  
Audience: PhoneBuddySDK and Tianyan Phone maintainers  
Last updated: 2026-08-29

## 1. Decision summary

This design makes the following decisions:

1. **PhoneBuddySDK owns routing mechanisms.** Retry, failover, cooldown,
   health scoring, group ordering, provider selection, state persistence,
   and route-affine protocol state are SDK responsibilities.
2. **The host app owns provider configuration and product policy.** Tianyan
   Phone continues to define server URLs, credentials, models, pool
   membership, base scores, grouping, workload bindings, and UI behavior.
   It passes a declarative configuration snapshot to the SDK; it does not
   run a second router.
3. **Main agent, subagents, and lightweight utility work use named provider
   pools.** The initial pool IDs are `main`, `subagent`, and
   `session_title`, but the model is intentionally generic so more utility
   workloads can be added without changing the router.
4. **Conversation-title generation uses a new generic SDK one-shot API.** It
   is not implemented as a subagent. The operation has no tools, task
   lifecycle, session mutation, agent loop, or persona. Tianyan Phone calls
   the one-shot API with the `session_title` pool and retains ownership of
   the title prompt, output cleanup, and local fallback.
5. **Local conversation items remain the source of truth.** Server response
   IDs and encrypted reasoning artifacts are optional, route-affine
   optimizations. They must never be required to recover or continue a
   conversation.

## 2. Background

Tianyan Phone and PhoneBuddySDK currently split LLM reliability behavior
across two layers:

- Tianyan Phone defines a server list and implements persisted failure
  scoring, group ordering, and chain construction in TypeScript.
- PhoneBuddySDK receives the resulting primary/fallback chain and implements
  request retry, error classification, provider switching, and an in-memory
  exponential cooldown.
- Tianyan Phone normally creates a new SDK engine for an active run. SDK
  cooldown state therefore does not reliably survive the next user turn,
  while the app's score state does.
- SDK provider events currently identify a route using a sanitized
  `host/model` fingerprint, while app health state is keyed by the app's
  endpoint ID. This makes failure attribution fragile.
- SDK subagents share the main agent's `LlmClient`, including its provider
  chain and mutable transport state.
- Conversation-title generation bypasses PhoneBuddySDK and directly builds
  backend-specific HTTP requests in the app. It consequently duplicates
  protocol adapters, retry/fallback behavior, headers, diagnostics, and
  credential handling.

This arrangement worked while there was one primary workload and a short
fallback list. It becomes ambiguous once main agents, concurrent subagents,
and low-cost utility generation require different server lists.

## 3. Problems to solve

### 3.1 Two competing routing state machines

The app's score window and the SDK's cooldown can disagree. The chain may be
sorted using one view of provider health and then changed again using another
view. A provider can also appear healthy after an SDK engine is recreated
even though it failed seconds earlier.

There must be one owner of runtime health and selection. Since retry and
failover depend on transport-level facts such as HTTP status, partial stream
state, `Retry-After`, backend compatibility, and response headers, that owner
must be PhoneBuddySDK.

### 3.2 Configuration is not the same as routing mechanism

Moving the mechanism into the SDK does not mean hard-coding Tianyan's server
catalog or cost policy in the SDK. URL, credential, model choice, base
priority, workload assignment, and product fallback behavior are app policy.
The SDK must execute that policy consistently and safely.

### 3.3 Workload isolation

The main agent may need a capable model and hosted tools. A subagent may use a
different cost/latency balance but still needs a multi-turn tool loop. A
session title needs only a short, inexpensive text completion and must not
gain agent powers accidentally.

One shared, implicit provider list cannot express these requirements.

### 3.4 Direct HTTP utility calls bypass SDK guarantees

The current title path must know how to construct each backend's payload and
headers. Every future utility task would repeat the same mistake. The SDK
needs a small non-agent generation surface that reuses its routing,
transport, wire adapters, diagnostics, and cancellation behavior.

### 3.5 Route-affine state can leak across concurrent work

Headers such as `x-codex-turn-state`, response IDs, and encrypted reasoning
are valid only within well-defined route and lifecycle boundaries. A main
agent and concurrent subagents may share long-lived SDK objects, so mutable
state stored directly on a transport can cross logical turns or race between
agents.

## 4. Goals and non-goals

### Goals

- One SDK router for every LLM workload in the process.
- Health and cooldown survive per-run engine recreation and, optionally,
  process restart.
- Stable app-supplied provider IDs are used consistently in state and events.
- Independent, named provider pools for main agent, subagents, and utilities.
- A generic, tool-free one-shot text API suitable for session titles and
  future lightweight tasks.
- Deterministic group/score ordering and deterministic exhaustion behavior.
- Safe handling of concurrent agents and provider-bound protocol artifacts.
- Backward-compatible migration from `primary + fallback_providers`.

### Non-goals

- The SDK will not choose Tianyan's vendors, credentials, prices, or default
  model tiers.
- The SDK will not own title wording, localization, title cleanup, or when a
  title should be generated.
- One-shot generation will not execute tools or become a second agent API.
- Server-stored response state will not replace persisted local conversation
  history.
- This design does not require background health probes. Passive recovery on
  a later operation is sufficient for the first implementation.

## 5. Ownership boundary

| Concern | Tianyan Phone | PhoneBuddySDK |
|---|---|---|
| Server URLs, API keys, custom headers/body | Defines and supplies | Validates and consumes |
| Models and client/backend profiles | Chooses | Adapts to wire protocol |
| Pool membership and workload binding | Chooses | Resolves by pool ID |
| Base score, stable order, routing group | Chooses | Applies deterministically |
| Retry classification and backoff | Observes events | Owns |
| Failure penalties and cooldown | Does not calculate | Owns and persists |
| Group ranking and provider selection | Does not sort | Owns |
| Response ID/header lifecycle | Does not manage | Owns |
| Encrypted-reasoning compatibility | Declares verified compatibility when needed | Enforces stripping/retention |
| Title prompt, cleanup, fallback title | Owns | Treats as opaque input/output |
| HTTP dumps and provider telemetry | Exports/displays if desired | Produces safely |

The app may inspect router telemetry for UI and diagnostics, but it must not
feed every SDK switch into a second app-side scoring algorithm after cutover.

## 6. Target architecture

```text
Tianyan provider catalog and product policy
                    |
                    v
        PhoneBuddyRuntime (long-lived)
        +-----------------------------+
        | LlmRouter                   |
        | - named pools               |
        | - shared health store       |
        | - retry/failover/cooldown   |
        | - transport/protocol state  |
        +-----------------------------+
             |          |          |
             v          v          v
        main pool   subagent pool   utility pool(s)
             |          |          |
       Agent Engine   TaskManager   generate_text()
       + tool loop    + tool loop   no tools/session
```

`PhoneBuddyRuntime` is longer-lived than an individual agent run. Engines and
one-shot calls borrow routing services from the runtime. This is necessary
because Tianyan may recreate an engine between user turns while provider
health must remain continuous.

The SDK should also persist a versioned health snapshot under its existing
root directory so useful cooldown state survives process restart. The
in-process runtime remains authoritative while the process is alive.

## 7. Configuration model

The exact Rust/FFI spelling may follow existing conventions, but the semantic
model must contain the following concepts.

```rust
struct LlmRoutingConfig {
    providers: Vec<ProviderTarget>,
    pools: BTreeMap<String, ProviderPool>,
    health: RouterHealthConfig,
}

struct ProviderTarget {
    provider_id: String,
    base_url: String,
    api_key: SecretString,
    model: String,
    api_backend: ApiBackend,
    client_profile: ClientProfile,
    reasoning_compatibility_key: Option<String>,
    capabilities: ProviderCapabilities,
    extra_headers: Map<String, String>,
    extra_body: Map<String, JsonValue>,
}

struct ProviderPool {
    members: Vec<PoolMember>,
    retry: RetryPolicy,
    when_exhausted: ExhaustionPolicy,
}

struct PoolMember {
    provider_id: String,
    routing_group: String,
    base_score: i32,
    order: u32,
    enabled: bool,
}
```

Example app-owned configuration:

```json
{
  "providers": [
    {
      "provider_id": "hermes-grok-main",
      "base_url": "https://hermes.example/v1",
      "api_key": "...",
      "model": "grok-4.6",
      "api_backend": "responses",
      "client_profile": "grok_build",
      "reasoning_compatibility_key": "grok-build/grok-4.6"
    },
    {
      "provider_id": "light-title-primary",
      "base_url": "https://light.example/v1",
      "api_key": "...",
      "model": "light-model",
      "api_backend": "chat_completions",
      "client_profile": "default"
    }
  ],
  "pools": {
    "main": {
      "members": [
        {"provider_id":"hermes-grok-main","routing_group":"preferred","base_score":10,"order":0,"enabled":true}
      ],
      "when_exhausted": "probe_earliest"
    },
    "subagent": {
      "members": [
        {"provider_id":"hermes-grok-main","routing_group":"default","base_score":10,"order":0,"enabled":true}
      ],
      "when_exhausted": "probe_earliest"
    },
    "session_title": {
      "members": [
        {"provider_id":"light-title-primary","routing_group":"cheap","base_score":10,"order":0,"enabled":true}
      ],
      "when_exhausted": "fail_fast"
    }
  }
}
```

### 7.1 Stable provider identity

`provider_id` is supplied by the app and must remain stable across config
reloads and process restarts. It identifies one concrete routable target,
including the endpoint, credential/quota identity, backend, and model. It
must not contain a secret.

Health is shared across pools by `provider_id`. If two workloads should have
independent quotas or health, the app must declare two provider IDs even if
their URLs happen to match. This makes sharing an explicit policy choice.

### 7.2 Do not overload `group`

Two independent concepts must have different fields:

- `routing_group` controls group ranking and fallback order inside a pool.
- `reasoning_compatibility_key` states that encrypted reasoning artifacts are
  safe to replay on another target.

Routing affinity does not prove cryptographic or model compatibility. The SDK
must default a missing compatibility key conservatively to the unique
`provider_id`. Tianyan should assign a shared key only to gateways whose
backend profile and exact model compatibility have been verified.

### 7.3 Pool aliases and backward compatibility

Pool inheritance must be explicit in the resolved configuration, for example
`subagent -> main`. For compatibility, the legacy config adapter may
synthesize:

- a `main` pool from the current primary and `fallback_providers`;
- a `subagent` alias to `main` when no subagent pool exists;
- no implicit `session_title` alias.

Missing utility pools return `RouteNotConfigured`. They must not silently use
an expensive main-agent model unless the app explicitly configures that
alias.

## 8. Router behavior

### 8.1 Health record

The SDK keeps a health record keyed by `provider_id`:

```rust
struct ProviderHealthRecord {
    recent_failures: Vec<Timestamp>,
    consecutive_trips: u32,
    cooldown_until: Option<Timestamp>,
    last_success_at: Option<Timestamp>,
    last_failure_class: Option<FailureClass>,
}
```

The first implementation should preserve Tianyan's existing scoring intent:

```text
effective_score = base_score - failures_within(penalty_window)
```

Recommended initial `penalty_window` is one hour. A successful operation
clears consecutive-trip cooldown escalation but does not erase historical
failures immediately; those penalties age out naturally. Retry attempts
inside one provider visit count as one trip, not multiple score penalties.

### 8.2 Deterministic ordering

For every operation:

1. Remove disabled members and reconcile expired health timestamps.
2. Prefer members with `effective_score > 0` that are not cooling.
3. Rank each routing group by the maximum effective score among its eligible
   members.
4. Break group ties by the lowest declared member `order`.
5. Within a group, order members by effective score descending, then `order`
   ascending.
6. Visit providers using the resulting deterministic sequence, applying the
   pool's retry budget before failover.

If no normally eligible member exists:

- `probe_earliest` selects the enabled provider whose suppression/cooldown
  expires first, with effective score and declared order as tie-breakers;
- `fail_fast` returns `PoolExhausted` with a retry-after hint.

This distinction lets the main agent preserve availability while cheap,
best-effort title generation can fail quickly.

### 8.3 Failure and recovery

Existing transport-aware classifications remain in the SDK. In particular:

- context overflow and `x-should-retry: false` terminate the operation and do
  not poison or switch the provider;
- retryable connection/5xx failures spend the in-provider budget, then trip;
- a long `Retry-After` may trip immediately and set a longer cooldown;
- a failure after visible streaming is not blindly replayed to another
  provider;
- success resets consecutive cooldown escalation and records the selected
  provider.

Cooldown progression may retain the current 120, 240, 480, 600 second shape,
but values belong in SDK policy configuration rather than app code.

### 8.4 Persistence and reconciliation

Persist only non-secret health data, using a versioned file such as:

```text
<root_dir>/.phonebuddy/router/health-v1.json
```

Requirements:

- write through a temporary file and atomic rename;
- serialize updates behind one runtime lock;
- discard corrupt or unsupported snapshots safely;
- remove records for provider IDs absent from the current config after a
  bounded retention period;
- prune expired failure timestamps before selection and persistence;
- never persist URLs, credentials, request bodies, prompts, response IDs,
  encrypted reasoning, or `x-codex-turn-state`.

A config update is reconciled by stable ID. Changing the meaning of a target
requires a new `provider_id`; otherwise stale health would intentionally
follow it.

## 9. Workload semantics

| Property | Main agent | Subagent | One-shot utility |
|---|---|---|---|
| Default pool | `main` | `subagent` (legacy alias to `main`) | Caller-supplied, e.g. `session_title` |
| Multi-turn tool loop | Yes | Yes | No |
| Tools | Full configured set | Restricted subagent set | Always none |
| Session read/write | Yes | Task-owned history | None |
| Compaction/persona | Yes | Subagent prompt rules | None |
| Background execution | Host-controlled | Supported | Not a task; async/cancellable call only |
| Typical model policy | Capable | Independently configurable | Cheap/fast |

### 9.1 Main agent

The engine resolves the `main` pool at the start of each logical user turn.
All LLM hops in its tool loop share one operation context, while the router
may fail over according to the pool policy.

### 9.2 Subagents

`TaskManager` must receive a client/router binding for the `subagent` pool
rather than implicitly cloning the main client's chain. Every spawned
subagent creates its own logical-turn context. Concurrent subagents therefore
share provider health but never share response IDs or sticky transport
headers.

The current free-form subagent `model_override` must not bypass pool routing.
It should either be removed from the public surface or resolved only against
an app-declared, allow-listed pool member/model variant.

### 9.3 One-shot generation

Add a generic SDK API, conceptually:

```rust
struct GenerateTextRequest {
    pool_id: String,
    instructions: Option<String>,
    input: String,
    max_output_tokens: Option<u32>,
    temperature: Option<f32>,
    reasoning_effort: Option<ReasoningEffort>,
    response_format: Option<ResponseFormat>,
    timeout_ms: Option<u64>,
}

struct GenerateTextResult {
    text: String,
    usage: Option<Usage>,
    provider_id: String,
    model: String,
    attempts: u32,
}

async fn generate_text(
    &self,
    request: GenerateTextRequest,
    cancellation: CancellationToken,
) -> EngineResult<GenerateTextResult>;
```

Mandatory behavior:

- build a normal backend-neutral `ConversationRequest` and reuse the router,
  adapters, retry/failover, HTTP dumps, and usage parsing;
- expose no tool definitions and force tool choice to none;
- do not create/load/save an agent session;
- do not run compaction, agent prompts, task state, or a tool loop;
- collect the streamed transport response internally and return one result;
- use a fresh logical operation context and never accept a
  `previous_response_id` from the app;
- support timeout and cancellation independently of an agent session ID.

This is a new SDK capability, not a title-specific API. Naming it
`generate_session_title` would put product policy in the wrong layer.

### 9.4 Why one-shot is not a subagent

A subagent is the wrong abstraction for a title because it intentionally
provides a prompt/persona, task record, multi-turn loop, tool surface,
resumption rules, and potentially multiple LLM calls. Disabling those pieces
case by case would create a hidden second one-shot API inside `TaskManager`.

The dedicated generic one-shot path shares only the primitives a title needs:
provider-pool selection, protocol adaptation, retry/failover, cancellation,
diagnostics, response collection, and usage metadata. It therefore guarantees
one tool-free generation operation and makes the lightweight server pool an
ordinary routing-policy choice.

## 10. Tianyan session-title flow

After cutover, Tianyan Phone should:

1. Decide when the current conversation has enough content for a title.
2. Build the localized title prompt in the app.
3. Call SDK `generate_text` with `pool_id = "session_title"`, a small output
   budget, and low/no reasoning effort.
4. Clean quotes/prefixes/line breaks and enforce title length in the app.
5. Save the title through the existing app data path.
6. On `RouteNotConfigured`, timeout, cancellation, or generation failure,
   retain the current deterministic local title fallback. Do not fail or
   interrupt the chat.

The existing direct Axios/backend-specific title request should then be
deleted. Tianyan retains only prompt construction, result sanitization, and
fallback behavior.

The `session_title` pool can contain its own ordered list of lightweight
servers. Its failures update SDK health like every other pool, but because it
usually uses distinct provider IDs they do not affect main-agent routing.

## 11. Response IDs, sticky headers, and encrypted reasoning

### 11.1 Source of truth

Persisted local conversation items are always sufficient to issue a full
request. Server-side response state is an optimization and may disappear at
any time.

### 11.2 `previous_response_id`

Normal main/subagent turns replay local history and do not carry a response
ID from a previous user turn. The current Responses SSE idle-timeout recovery
may reuse a captured response ID once on the exact same provider to continue
the same logical LLM operation. It must be cleared on provider switch,
backend/profile change, operation end, or connection/session invalidation.

A future WebSocket incremental path may use a response ID only when it can
prove the same connection, provider, request properties, and strict history
prefix. Full replay remains the fallback.

### 11.3 `x-codex-turn-state`

This header is scoped to one logical main-agent/subagent turn and one exact
transport. It may be reused across retries and tool-loop LLM hops within that
turn. It must not cross user turns, providers, one-shot calls, or concurrent
subagents.

The immediate correctness fix for this invariant is to carry an explicit
turn context from `LlmClient::begin_turn()` into transports instead of storing
the token globally on `HttpTransport`.

### 11.4 Encrypted reasoning

Text, tool calls, and tool outputs survive provider switches. Encrypted or
signed reasoning artifacts survive only when source and target have the same
explicit `reasoning_compatibility_key`. Otherwise the SDK strips reasoning
IDs, encrypted content/signatures, and plain reasoning content before replay.

## 12. SDK and native API shape

The implementation should introduce an additive long-lived runtime handle:

```rust
let runtime = PhoneBuddyRuntime::new(routing_config, root_dir)?;
let engine = runtime.create_engine(agent_config, "main")?;
let title = runtime.generate_text(title_request, cancellation).await?;
```

Recommended FFI concepts:

- `pb_runtime_new(routing_config_json, root_dir)`
- `pb_runtime_update_routing(runtime, routing_config_json)`
- `pb_engine_new_with_runtime(runtime, agent_config_json, main_pool_id)`
- `pb_runtime_generate_text_async(runtime, request_json, callback)`
- `pb_runtime_cancel_operation(runtime, operation_id)`
- `pb_runtime_free(runtime)`

The existing `pb_engine_new` path remains as a compatibility adapter that
creates a private runtime and synthesizes a legacy `main` pool. Mobile
wrappers should expose the same runtime/operation lifecycle without leaking
Rust implementation details.

One-shot completion callbacks should return a versioned JSON envelope and a
stable operation ID. They should not reuse chat session callbacks because a
utility request has no session lifecycle or agent events.

## 13. Events and diagnostics

New or versioned routing events should include:

```json
{
  "operation_id": "op_...",
  "workload": "one_shot",
  "pool_id": "session_title",
  "from_provider_id": "light-title-primary",
  "to_provider_id": "light-title-backup",
  "failure_class": "retryable_http",
  "cooldown_ms": 120000
}
```

Requirements:

- use stable `provider_id`, not `host/model`, as the state/event join key;
- a separate sanitized label may be included for human diagnostics;
- never include API keys or unmasked authorization headers;
- report a provider trip once per logical visit, not once per low-level retry;
- include `pool_id`, workload kind, operation ID, and selected provider in
  one-shot results and diagnostics;
- preserve legacy event fields during a deprecation window if the app UI
  depends on them.

## 14. Concurrency invariants

1. Router health is shared and synchronized across main, subagent, and
   one-shot operations.
2. Each logical main-agent turn gets a new operation context.
3. Each subagent gets its own operation context, even when spawned by the same
   parent turn.
4. Each one-shot call gets its own operation context.
5. Route-affine ephemeral state is additionally keyed by exact provider
   transport.
6. Applying the same provider trip twice for one operation/visit is
   idempotent.
7. Configuration replacement and selection observe a coherent versioned
   snapshot; an in-flight operation may finish on its captured snapshot.

## 15. Error model

Add stable error kinds that wrappers can distinguish without parsing text:

- `InvalidRoutingConfig`
- `RouteNotConfigured { pool_id }`
- `PoolExhausted { pool_id, retry_after_ms }`
- `ProviderAttemptsExhausted { pool_id, tried_provider_ids }`
- `OperationTimedOut`
- `OperationCancelled`
- existing transport/backend errors with sanitized metadata

For title generation, Tianyan treats all of these as best-effort failures and
uses its local fallback. Main and subagent flows retain their existing
user-visible error handling.

## 16. Migration plan

### Phase 0: fix unsafe ephemeral state

- Replace transport-global `x-codex-turn-state` with logical-turn-scoped
  context.
- Start one context per main turn and per subagent.
- Add same-turn reuse, cross-turn isolation, and concurrent-session tests.

This phase is independent and should land before the router migration.

### Phase 1: add IDs, pools, and runtime router behind a legacy adapter

- Add routing config types and validation.
- Implement deterministic group/score selection and SDK health persistence.
- Introduce `PhoneBuddyRuntime` and stable provider-ID events.
- Convert old `primary + fallback_providers` into a synthesized `main` pool
  so existing callers continue to work.

### Phase 2: bind agent workloads

- Resolve the main engine through `main`.
- Give `TaskManager` an explicit `subagent` pool binding/client.
- Preserve legacy `subagent -> main` inheritance only in the adapter.
- Remove or constrain model override behavior that bypasses pools.

### Phase 3: add one-shot core and FFI

- Implement `generate_text` without agent/session/tool dependencies.
- Add async native entry points, cancellation, result metadata, and wrapper
  bindings.
- Test every supported backend adapter through the common request model.

### Phase 4: move Tianyan configuration policy

- Keep the raw provider catalog in `cloudAgentConfig.local.ts` and related
  app config.
- Add named pool definitions and stable provider IDs.
- Stop sorting providers and calculating failure scores in TypeScript.
- Pass the declarative routing snapshot to the long-lived SDK runtime.
- Change UI/event consumers to join on `provider_id`.

### Phase 5: migrate session-title generation

- Configure the `session_title` lightweight pool.
- Replace direct HTTP with SDK `generate_text`.
- Keep app prompt, cleanup, length enforcement, and deterministic fallback.
- Delete backend-specific Axios payload construction after rollout proves
  equivalent behavior.

### Phase 6: remove duplicate state

- Remove app-side provider priority persistence and `recordFailure` calls.
- Delete compatibility event joins based on `host/model`.
- Update existing fallback documentation and operational runbooks.

Do not run both app scoring and the new SDK scoring permanently. A short
telemetry-only shadow period is acceptable, but only the SDK result may drive
selection.

### Existing app health migration

Provider penalties are short-lived operational data. The recommended cutover
is a versioned reset rather than importing AsyncStorage records whose keys do
not reliably match SDK fingerprints. If continuity is later required, import
only records that map unambiguously to the new stable `provider_id`.

## 17. Acceptance matrix

| Scenario | Expected result |
|---|---|
| Main primary returns retryable 503 | SDK retries within budget, switches once, records one trip, emits stable IDs |
| New engine is created seconds later | Shared runtime/disk health starts on the healthy backup |
| Cooldown and failure window expire | Primary becomes eligible deterministically and can recover on success |
| Providers share a routing group | Group rank is max member score; member and tie order are stable |
| All main providers are cooling | `probe_earliest` chooses the deterministic earliest candidate |
| All title providers are unavailable | `fail_fast` returns promptly; app uses local title fallback |
| Main and two subagents run concurrently | Health is shared; response IDs and turn-state headers are isolated |
| Subagent has a separate cheap pool | It never silently calls the main pool |
| Same provider ID appears in two pools | Its trip affects both pools by design |
| Separate IDs use the same URL | Their health remains independent |
| Provider switches mid-history | Text/tools survive; incompatible encrypted reasoning is stripped |
| Idle timeout continues on same Responses provider | Same-operation response ID may be used once; failover clears it |
| A new user turn begins | No prior `previous_response_id` or `x-codex-turn-state` is sent |
| Title generation succeeds | No tools/session files/tasks are created; result reports provider and usage |
| Title pool is not configured | SDK returns `RouteNotConfigured`; app keeps deterministic fallback |
| App updates pool config | New operations use one coherent version; retained provider IDs keep health |
| Process restarts | Versioned non-secret health reloads; corrupt state fails open safely |
| HTTP dump/event inspection | Credentials and ephemeral routing tokens are masked; router health persists none of them |

## 18. Required test layers

### SDK unit tests

- score decay, cooldown escalation/recovery, group ordering, and tie breaks;
- config reconciliation by stable provider ID;
- trip idempotence within one operation;
- compatibility-key sanitization;
- logical-turn transport-state isolation;
- one-shot request construction always excludes tools and sessions.

### SDK integration tests

- scripted multi-provider failover for each error class;
- concurrent main/subagent/one-shot calls against shared health;
- real local HTTP server assertion for header and response-ID lifecycle;
- health persistence/reload and corrupt-file recovery;
- FFI async completion and cancellation ownership.

### Tianyan integration tests

- config-to-SDK pool serialization with no secrets in logs;
- provider event mapping by stable ID;
- title success, timeout, missing pool, malformed output, and local fallback;
- engine recreation does not reset routing health;
- removal of TypeScript scoring does not change banner/error behavior.

## 19. Implementation completion criteria

The migration is complete only when:

1. Every main/subagent/utility LLM request passes through the same SDK router.
2. Tianyan supplies policy but contains no active retry/cooldown/score router.
3. Main, subagent, and title pools can be configured independently.
4. Session-title generation has no direct app HTTP/backend payload code.
5. Route-affine ephemeral state passes cross-turn and concurrency tests.
6. Provider events and persisted health use stable app-supplied IDs.
7. The legacy config path remains tested for existing SDK consumers.
8. SDK checks/tests and Tianyan unit/E2E tests pass on the supported mobile
   platforms.

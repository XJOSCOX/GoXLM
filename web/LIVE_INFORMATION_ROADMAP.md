# GoXLM roadmap for live information and memory

GoXLM will support natural conversation through evaluated, independently
trained checkpoints. Live information will come from an opt-in retrieval
service. The existing local Web interface, completion endpoints and checkpoint
allowlist remain the starting point. The conversation formatter and future
checkpoint admission rules are documented in [CONVERSATION.md](CONVERSATION.md).

## Request flow and permissions

1. The local UI submits the current question. A deterministic server-side router
   chooses ordinary conversation, web search, news, market data, or a
   clarification. It uses explicit freshness cues and typed entities. “What's
   today's news?” routes to news using the user's time zone. “How is NVIDIA
   stock doing?” routes to quotes after resolving the listing and currency.
   “How does a transistor work?” stays local unless the user asks for current
   sources. An ambiguous symbol or date triggers a clarification.
2. A separate policy gate checks that external retrieval is enabled, that the
   specific provider and tool are permitted, and that the outbound query fits
   the grant. Start with per-request confirmation that previews the provider
   and exact outbound query. Cancel or deny leaves the request local. Treat
   routing as a recommendation even if a future learned router replaces it.
   Only the active question or a minimal derived query may leave the machine.
   Keep saved memories, raw conversation history, private identifiers and
   local file paths out of provider requests.
3. A restricted retrieval worker runs typed, read-only provider operations with
   server-held credentials and strict budgets. The orchestration layer validates
   normalized results and assembles source-linked evidence. Render headlines and
   quotes directly from validated fields. When an evaluated conversation
   checkpoint can summarize evidence, give it a bounded set of untrusted
   excerpts and validate its citation IDs against that exact set. Show the
   original sources and flag failed or unsupported summaries. The current
   completion checkpoint continues through the existing inference path.
4. Stream results and structured errors to the local UI. Cancel queued fetches
   and in-flight HTTP requests when a request is stopped or disconnected.
   Track timeouts, result ages, and quota usage without logging prompts,
   fetched bodies, or provider secrets. Provider errors, disabled access,
   and missing evidence produce an explicit unavailable state.

## Adapter contracts and freshness

Version provider schemas independently of the conversation format. Common
fields include provider and source IDs, canonical URL where available,
retrieval time in UTC, source timestamp and its meaning, expiry, and
cache/display permissions. Keep source publication or observation time
distinct from retrieval time. Display the user's time zone with an explicit
label. Use provider terms and measured behavior to set initial TTLs and rate
limits. The effective limit is the strictest applicable provider, product,
or user rule. Expired results may appear only where retention and display
rights permit and with a conspicuous age and stale label.

| Adapter | Typed input and required result |
|---|---|
| Search and page retrieval | Accept a bounded query. Return result IDs and ranked links from an approved provider. Fetch a cited result through the restricted worker. Record the final URL, publisher, title, retrieval time, and source date when available. Distinguish publication from page modification. Set trusted domains per use case and retain source identity without treating page text as instructions. |
| News | Accept region, language and time window. Return major headlines with publisher, publication date and time, source link and retrieval time. Collapse duplicate coverage without hiding publisher attribution. A cached copy or provider refresh cannot move an old headline into today's window. |
| Markets | Resolve instrument ID, symbol, exchange and currency. Return last price, daily absolute and percentage change, reference close, session type/status, quote timestamp, exchange time zone and entitlement/delay classification. Keep premarket, regular and after-hours quotes distinct. Use a licensed feed with confirmed display, cache, redistribution and attribution rights. Label real-time, delayed, end-of-day or stale only from validated entitlement and timing metadata. If freshness is unknown, show the timestamp and unverified status. Market hours alone do not establish live pricing. |
| Weather, publications and other services | Add adapters under separate review. Weather needs location, observation or forecast issue time, valid time and units. Publications need bibliographic ID, version, publication date, retrieval time and correction or retraction status where available. |

Use meaningful freshness rules per field. News windows depend on the requested
date and location. Market freshness depends on session, exchange calendar,
quote type, feed entitlement and source time. A closed session can display the
latest licensed close with its date and closed status. Missing quotes, rates
or timestamps stay missing. Never synthesize a price or imply last close is
a current trade. Define entitlement and cache behavior before selecting a
financial provider or enrolling in a paid plan.

## Separate data stores

Use distinct namespaces and access paths, with local storage outside Git.
Persist only the minimum policy metadata needed to apply expiry and deletion.
Default to memory-only retrieval and no provider-content cache until each
adapter has a retention rule.

| Store | Contents and lifecycle |
|---|---|
| Private conversation history | User and assistant messages, their roles and timestamps, and optional citation references. Preserve the browser-local history experience. Any future migration to a local database is user-controlled and versioned. Provide deletion and bound persistence. |
| Persistent user memory | User-confirmed preferences or facts with provenance, scope, timestamps, optional expiry and an edit history subject to retention limits. Provide inspect, edit, disable and delete controls. Retrieve a small relevant subset under the tokenizer budget. Invalidate dependent indexes and summaries on change or deletion. |
| Temporary retrieval | Request-scoped search results, excerpts and provider payloads. Expire on request completion by default and clean up on cancellation. Where a provider forbids caching, retain only what is needed to render the response in the active session and prevent saved history from retaining provider payloads. |
| Source-linked factual cache | Only provider-permitted normalized facts or snippets, with evidence IDs, observation and retrieval times, expiry and scope. Store changes as versions linked to the original source. Revalidate or mark expired entries before use and purge cached content after source or permission changes. Keep query keys and document scope isolated so a cache hit cannot reveal another user's data. |
| Rights-approved training data | Separate quarantine, evidence, reviewer decisions and immutable release manifests. Access is restricted to the offline review and training workflow. No ingestion path from conversation, memory, retrieval or fact caching directly into this store. |

Using history, memory, or citations changes inference context. Neural-network
weights change only in an authorized offline training run.

## Retrieval security and privacy

- Keep the UI and API bound to loopback. Authenticate local privileged setup,
  enforce exact trusted origins and Host headers, and protect state-changing
  requests against cross-site requests. External browsing starts disabled.
  Keep provider credentials out of model context, frontend bundles and logs.
- Restrict the fetch worker's egress outside its process. Begin with approved
  query APIs and allowlisted public page hosts. Accept only canonical HTTPS
  URLs and approved ports, reject embedded credentials, and reject private,
  loopback, link-local, metadata, multicast and reserved IPv4/IPv6 targets.
  Check resolved addresses, connect only to a checked address while verifying
  the original hostname and certificate, and recheck every redirect. Reject
  proxy bypasses, disguised addresses and resolution changes. A blocked
  redirect fails closed. The worker should have no route to local service ports.
- Keep external text in a typed evidence channel. Strip active page content,
  do not run scripts or macros, render escaped text and safe links, and bound
  bytes, decompression, redirects, extraction work and model input. Evidence
  cannot grant tool permission, carry privileged role markers, read local
  memory or select a second outbound action. Apply the same rules to search
  snippets, headlines and provider error bodies.
- Enforce per-request call and byte budgets, timeouts, bounded retry and
  concurrency limits, provider quotas, and a local circuit breaker. Make
  cross-provider fallback visible and subject to the user's approval.
  Exercise quota exhaustion and cancellation with synthetic response fixtures.
- Record provider retention and downstream disclosure before enabling a
  tool. Store minimal local operational metadata. Test expiration, deletion,
  cache invalidation, backups, and recovery after an interrupted cleanup.

## Reviewed continuous improvement

An explicit contribution opt-in applies to selected records and purposes.
Ratings are feedback and corrections are candidate records. Quarantine
contributions, redact private details, verify provenance and commercial
training permission, and screen for low quality, duplication and poisoning.
Rights reviewers and quality reviewers record separate decisions against the
exact retained text and its evidence. Apply the existing approval and
revocation workflow. Snapshot eligible items in a versioned release with
family-level split isolation, and record release, tokenizer, training run,
parent checkpoint and output checkpoint hashes.

Withdrawal or revocation blocks future inclusion, triggers deletion or
restricted retention for the affected records, and traces impacted releases
and descendant checkpoints. Keep minimal audit identifiers where required.
Review affected checkpoints for quarantine, replacement or retraining from
a clean lineage. A deletion operation on stored text cannot erase influence
from existing weights. Compare any replacement on held-out conversation,
retrieval grounding and privacy evaluations before release.

## Implementation phases and acceptance gates

1. **Contracts and offline routing.** Test adapters and freshness using
   synthetic results with a controllable clock. Build a held-out routing set
   spanning greetings, timeless questions, explicit recency, ambiguous
   tickers, time zones and quote requests. Require zero outbound requests
   with browsing disabled or denied across all fixtures. Set a routing
   accuracy target before enabling the policy gate.
2. **Opt-in web and news.** After authorization, implement the worker and
   enable only approved providers. Gate on hostile redirects, DNS rebinding,
   private-address fixtures, injected instructions, stale results, cancellation
   and resource budgets. Require every displayed headline and evidence claim
   to resolve to a publisher, dated source or explicitly missing date, and
   the captured source link.
3. **Market data and future adapters.** Verify entitlements before provider
   activation. Test market holidays, delayed and stale feeds, unavailable
   quotes, exchange time zones, currency and session transitions. Require
   every fixture quote to render its source, timestamp and correct freshness
   label. Introduce weather and publications only after separate review.
4. **Conversation and memory.** Admit only evaluated dialogue checkpoints
   under the versioned formatter contract. Introduce user-controlled memory
   with retention and deletion tests, a history-only comparison, a
   context-truncation test, and an opt-in query preview. Add learned routing
   only if it beats deterministic routing on held-out cases while sharing
   the same permission gate.
5. **Reviewed learning.** Dry-run consent, redaction, review, withdrawal and
   lineage tracing with synthetic records. Require exact traceability from
   any included record to releases and checkpoints. Authorize each future
   corpus acquisition and training experiment separately.

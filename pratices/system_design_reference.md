# System Design Reference — One Line + Real Example

---

## What to Learn for System Design Interviews (ML Engineer Focus)

### Learn Deeply — You Will Be Asked These
These come up in every FAANG system design interview for ML/backend roles. Know them well enough to draw and explain on a whiteboard.

| Topic | Why Interviewers Ask |
|-------|---------------------|
| **Caching (Redis, TTL, cache-aside, write-through)** | "Design a prediction service" — cache is always the first optimization |
| **Message Queue (Redis queue, Kafka)** | "How do you handle 10K batch requests?" — async queue is the answer |
| **Rate Limiting (token bucket algorithm)** | "How do you protect your API?" — always asked for public-facing services |
| **Load Balancing + Horizontal Scaling** | "Your service gets 10× traffic" — add servers, distribute load |
| **SQL Indexing + Slow Query fixes** | "Your DB is slow" — missing index is #1 cause, always check this first |
| **CAP Theorem** | "Choose between consistency and availability" — every distributed system tradeoff question |
| **Pub/Sub + Webhooks** | "How does GitHub trigger your CI?" — event-driven architecture |
| **Blue-Green + Canary Deployment** | "How do you deploy without downtime?" — standard for any ML model update |
| **JWT vs Session Auth** | "How do you secure your API?" — stateless vs stateful tradeoff |
| **Kafka** | "Design a real-time feature pipeline" — Kafka is the industry answer |
| **Service Discovery (Kubernetes DNS)** | "How do microservices find each other?" — K8s handles this automatically |
| **Sharding** | "Your DB has 1 billion rows" — split by key across multiple servers |
| **Circuit Breaker** | "What if a downstream service is slow?" — fail fast, don't cascade |

### Know the Concept — Explain in 2 Sentences
You won't implement these but interviewers expect you to know what they are.

| Topic | What to Say |
|-------|-------------|
| **Consistent Hashing** | "Distributes keys across nodes on a ring — adding a node only remaps a fraction of keys, not all" |
| **WAL (Write-Ahead Log)** | "Postgres writes changes to a log before applying them — crash recovery replays the log" |
| **Event Sourcing** | "Store every change as an event, not just current state — full audit trail, replayable" |
| **CQRS** | "Separate read and write models — reads and writes scale independently" |
| **Saga Pattern** | "Chain of transactions across microservices — each step has a compensating rollback if it fails" |
| **Bloom Filter** | "Probabilistic data structure — 'definitely not in set' or 'maybe in set' — saves DB lookups" |
| **Thundering Herd** | "Cache expires → 1000 requests hit DB simultaneously — fix: one request rebuilds cache, others wait" |
| **Backpressure** | "Consumer tells producer to slow down when overwhelmed — prevents queue from exploding" |

### Skip for ML Engineering Interviews
Interviewers for ML Engineer roles rarely ask these. Focus your time elsewhere.

| Skip | Reason |
|------|--------|
| WebRTC internals | Only for video infrastructure teams |
| DNS / BGP internals | Network engineering, not ML |
| Web crawler design | Backend SWE, not ML |
| URL shortener from scratch | Classic SWE interview, rarely asked for ML roles |
| mTLS implementation | Security engineering specialty |
| IPv4/IPv6 | Infrastructure, not ML |
| Multi-master conflict resolution | DB internals, not ML |

### The 4 Questions That Define an ML Engineer Interview

Every FAANG system design round for ML engineers comes down to these:

1. **"How does your model get to production?"**
   → Training pipeline → MLflow gate → ONNX export → Docker → ECR → Helm → EKS

2. **"How do you know the model is degrading?"**
   → PSI drift detection → Prometheus metrics → Grafana → Slack alerts → ground truth MAE

3. **"How do you roll back a bad model safely?"**
   → Shadow → A/B → canary → if health gate fails → rollback to previous ONNX version

4. **"How do you handle 10× traffic?"**
   → Redis cache (40% hit rate) → ONNX no GIL → Gunicorn workers → K8s HPA (2→5 pods)

**This project answers all 4. Know these cold.**

---

## 1. Distributed Systems

| Topic | One Line | Real Example |
|-------|----------|--------------|
| **Replication** | Copy data to multiple nodes so if one dies, others serve it | Postgres primary + 2 read replicas — writes go to primary, reads spread across all 3 |
| **Heartbeat** | Node sends "I'm alive" ping every N seconds — no ping = dead | Kubernetes kubelet pings control plane every 10s. No ping for 40s → pod marked dead |
| **Leader Election** | Nodes vote to pick one coordinator for writes | ZooKeeper elects one Kafka broker as controller. Others follow it |
| **Split-Brain** | Two nodes both think they're the leader after a network partition — data diverges | Two data centers lose connection. Both accept writes. When reconnected — conflicting data. Solved by quorum |
| **Quorum** | Need majority (N/2 + 1) of nodes to agree before committing a write | 3 nodes: need 2 to agree. 5 nodes: need 3. Prevents split-brain |
| **WAL (Write-Ahead Log)** | Write the change to a log file BEFORE applying it — crash recovery replays the log | Postgres writes every change to WAL first. Server crashes → replays WAL on restart, no data lost |
| **Circuit Breaker** | If a downstream service fails 5× in a row, stop calling it for 30s instead of hammering it | Payment service is down. Circuit breaker opens → return "service unavailable" immediately instead of waiting 30s per request |
| **Thundering Herd** | Cache expires → 1000 requests simultaneously hit the DB | Redis key expires at midnight. 1000 users all query DB at once. Fix: cache lock + one request rebuilds, others wait |
| **Consistent Hashing** | Map both data and nodes onto a ring — node added/removed only moves a fraction of keys | Adding a 4th Redis node only remaps 25% of keys, not all of them |
| **Event Sourcing** | Store every state change as an event, not the current state | Bank account: store [deposited $100, withdrew $30] not just "balance: $70". Replay events = full audit trail |

---

## 2. Consistency & Availability

| Topic | One Line | Real Example |
|-------|----------|--------------|
| **CAP Theorem** | Distributed system can guarantee only 2 of: Consistency, Availability, Partition-tolerance | Postgres: CP (consistent + partition tolerant, may be unavailable during partition). Cassandra: AP (available always, eventually consistent) |
| **Strong Consistency** | Every read sees the latest write, no matter which node you ask | After writing to Postgres primary, any read on any replica sees that write immediately |
| **Eventual Consistency** | Reads may return stale data — but all nodes will converge eventually | DynamoDB: write in us-east-1, read in eu-west-1 may see old data for 200ms |
| **Consistency Patterns** | Write-through, write-back, read-through — strategies for keeping cache and DB in sync | See Caching section below |

---

## 3. Database

| Topic | One Line | Real Example |
|-------|----------|--------------|
| **Indexing** | Pre-sorted lookup structure so DB doesn't scan every row | `SELECT * FROM listings WHERE neighbourhood = 'Manhattan'` — without index: scan 20K rows. With index: jump straight to 3K Manhattan rows |
| **Slow Query** | Query takes too long — usually missing index, N+1 problem, or full table scan | `SELECT * FROM predictions WHERE DATE(created_at) = '2026-07-14'` — wrapping in DATE() prevents index use. Fix: range query instead |
| **Sharding** | Split one giant table across multiple DB servers by a key | User IDs 1-1M → DB server 1. User IDs 1M-2M → DB server 2. Each server handles a fraction |
| **Vertical Scaling** | Make the one server bigger (more RAM, more CPU) | Upgrade RDS from db.t3.medium to db.r6g.4xlarge — 10× more RAM, same single server |
| **Horizontal Scaling** | Add more servers instead of making one bigger | Add 3 more read replicas — 4 servers total handling read traffic |
| **Master-Slave** | One primary accepts writes, replicas copy from it and serve reads | MySQL: one master + 3 slaves. Writes to master, reads from any slave |
| **Multi-Master** | Multiple nodes accept writes — conflict resolution required | CockroachDB: write to any node in any region. Conflict resolution via consensus |
| **NoSQL** | Schema-free database — document, key-value, column, or graph | MongoDB: store listing as JSON blob, no fixed schema. Redis: key-value for cache |
| **Idempotency** | Same request sent twice produces the same result, no side effects | POST /payments with idempotency-key header — network retry won't charge twice |

---

## 4. Caching

| Topic | One Line | Real Example |
|-------|----------|--------------|
| **Write-Through** | Write to cache AND DB at the same time | User updates profile → write to Redis + Postgres simultaneously. Always consistent, slower writes |
| **Write-Around** | Skip cache on write, write directly to DB — cache filled on next read | Bulk import 1M rows — write to DB only. Cache loaded when users actually read those rows |
| **Write-Back** | Write to cache only, flush to DB asynchronously later | Write to Redis instantly, background job syncs to Postgres every 5s. Fast writes, risk of data loss on crash |
| **Cache Aside** | App checks cache first, on miss reads DB and populates cache | This project: Redis miss → ONNX inference → write to Redis → return price |
| **TTL (Time To Live)** | Cache entry auto-expires after N seconds | This project: Redis prediction cache expires in 5 minutes |
| **Cache Stampede** | Same as Thundering Herd — many misses hit DB simultaneously | Fix: mutex lock so only one request rebuilds the cache |

---

## 5. Networking & Protocols

| Topic | One Line | Real Example |
|-------|----------|--------------|
| **TCP vs UDP** | TCP: guaranteed delivery, ordered. UDP: fast, fire-and-forget | HTTP uses TCP. Video streaming uses UDP (dropping a frame is OK, latency is not) |
| **HTTP vs WebSocket** | HTTP: request-response (client always initiates). WebSocket: bidirectional persistent connection | Chat app: WebSocket — server can push messages. REST API: HTTP — client asks, server answers |
| **WebSocket vs SSE** | WebSocket: two-way. SSE (Server-Sent Events): server pushes only, simpler | Live stock prices → SSE (server pushes updates). Multiplayer game → WebSocket (both sides send) |
| **WebRTC** | Peer-to-peer browser communication — video/audio/data without a server relay | Google Meet: video goes directly browser-to-browser, not through Google's server |
| **gRPC vs REST** | gRPC: binary (Protobuf), strongly typed, 10× faster. REST: JSON, human-readable, universal | Internal microservice calls → gRPC. Public API → REST (every client understands JSON) |
| **GraphQL vs REST** | GraphQL: client specifies exact fields needed. REST: server decides what to return | GitHub API: `query { user { name, repos { name } } }` — get exactly name + repos, nothing else |
| **Forward Proxy vs Reverse Proxy** | Forward: sits in front of clients (VPN). Reverse: sits in front of servers (Nginx) | VPN = forward proxy (hides your IP). Nginx = reverse proxy (hides your server, handles SSL) |
| **IPv4 vs IPv6** | IPv4: 4 billion addresses (running out). IPv6: 340 trillion trillion trillion addresses | AWS still uses IPv4 internally. IPv6 for new public internet deployments |
| **DNS** | Translates domain names to IP addresses | `nyc-airbnb.com` → `54.23.145.12` — your browser asks DNS, gets IP, connects |
| **CDN** | Serve static files from servers close to the user | React build files served from Cloudflare edge in Mumbai → 20ms latency instead of 200ms from us-east-2 |
| **CORS** | Browser blocks cross-origin requests unless server explicitly allows it | Frontend on `localhost:3000` calling API on `localhost:8001` → CORS error. Fix: `Access-Control-Allow-Origin` header |

---

## 6. Authentication & Security

| Topic | One Line | Real Example |
|-------|----------|--------------|
| **Session-Based Auth** | Server stores session in memory/DB, sends session ID in cookie | Flask login: session stored in Redis, cookie has session ID. Stateful — server must remember you |
| **JWT (JSON Web Token)** | Server signs a token with a secret — client sends it with every request, server verifies signature | This project: `X-API-Key` header. JWT version: token contains `{user: "subhash", exp: "2026-08-01"}`, signed with HMAC-SHA256 |
| **OAuth** | "Login with Google" — delegate authentication to a trusted third party | "Sign in with GitHub" on a website — GitHub verifies identity, issues token to the app |
| **MFA (Multi-Factor Auth)** | Require two proofs of identity — password + phone code | AWS login: password + 6-digit TOTP code from Authenticator app |
| **Rate Limiting** | Limit how many requests a client can make in a time window | This project: `100/minute` per IP via `slowapi`. Exceeds → 429 Too Many Requests |
| **Rate Limit Algorithms** | Token bucket (burst allowed), sliding window (smooth), fixed window (simple) | Token bucket: start with 100 tokens, refill 10/sec. Each request costs 1 token. Can burst to 100 then trickle |
| **Trivy** | Scans Docker images for known CVEs in OS packages and libraries | CI: Trivy scans `nyc-airbnb:latest` before push to ECR. HIGH CVE found → build fails |
| **SonarQube** | Static code analysis — finds bugs, code smells, security vulnerabilities in source code | PR check: SonarQube finds hardcoded API key in config.py → blocks merge |

---

## 7. Architecture Patterns

| Topic | One Line | Real Example |
|-------|----------|--------------|
| **Monolith vs Microservices** | Monolith: one deployable. Microservices: many small independent services | This project = monolith (one FastAPI app). Uber = microservices (trip service, payment service, driver service — all separate) |
| **Pub/Sub** | Publisher sends messages to a topic, multiple subscribers receive them | Slack: you publish a message to #general topic. All channel members (subscribers) receive it |
| **Message Queue** | Producer puts jobs in queue, consumer picks them up asynchronously | This project: batch jobs in Redis queue. Worker picks one up via BRPOP, processes it, moves on |
| **Kafka** | High-throughput distributed message queue — millions of events/second, replay-able | Uber: every GPS ping from every driver published to Kafka. Multiple consumers (pricing, ETAs, fraud) all read independently |
| **Webhooks** | Server calls YOUR URL when an event happens — you don't poll | GitHub webhook: on every push, GitHub POSTs to your CI URL. Your CI starts the build |
| **SOLID Principles** | 5 OOP design rules: Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, Dependency Inversion | Single Responsibility: `predictor.py` only does inference. `store.py` only does DB writes. Not one God class |
| **Dependency Inversion** | High-level modules should not depend on low-level modules — both depend on abstractions | `api.py` depends on `PredictorInterface`, not `NYCAirbnbPredictorONNX` directly. Swap ONNX for Triton without changing api.py |
| **Bloom Filter** | Space-efficient probabilistic structure — "definitely not in set" or "probably in set" | Google Chrome: checks if a URL is malicious. If Bloom filter says "no" → safe (skip DB lookup). If "maybe" → check real DB |
| **Sticky Sessions** | Load balancer always routes same user to same server | Shopping cart stored in memory on server 1. Without sticky sessions, server 2 has no cart |
| **Load Balancing** | Distribute incoming traffic across multiple servers | Nginx: round-robin across 3 FastAPI pods. Each pod gets ~33% of requests |
| **Service Discovery** | Services find each other by name, not hardcoded IP | Kubernetes: `http://api-service:8001` — K8s DNS resolves to the right pod IP automatically |
| **Blue-Green Deployment** | Run two identical environments — switch traffic from blue (old) to green (new) instantly | Deploy v2 to green environment. Test it. Switch load balancer to green. v1 (blue) stays on standby for instant rollback |

---

## 8. Performance

| Topic | One Line | Real Example |
|-------|----------|--------------|
| **Debouncing** | Wait until user stops typing before firing the function | Search box: don't query API on every keystroke — wait 300ms after last keystroke |
| **Throttling** | Fire at most once per interval, regardless of how many triggers | Scroll event: update position at most once per 16ms (60fps), ignore intermediate events |
| **Indexing (Performance)** | Same as DB indexing — pre-computed lookup speeds up queries | Elasticsearch: inverted index on listing descriptions. Search "cozy Manhattan" → instant results |

---

## 9. Storage & Infrastructure

| Topic | One Line | Real Example |
|-------|----------|--------------|
| **S3** | Unlimited object storage — store any file, pay per GB | This project: ONNX model files stored in S3. CI downloads the latest champion at deploy time |
| **Lambda** | Serverless function — runs on demand, scales to zero, pay per execution | Monthly ground-truth job: Lambda runs once, downloads InsideAirbnb data, joins predictions, exits. No server running 24/7 |
| **Elasticsearch** | Distributed full-text search engine — indexes text for instant search | Log search in Datadog: `error AND neighbourhood:Manhattan AND latency:>500` — finds matching logs in milliseconds |
| **Cron Jobs** | Schedule a task to run at a fixed time/interval | This project: nightly drift check at 2am via GitHub Actions schedule: `0 2 * * *` |
| **Nginx** | Reverse proxy, load balancer, static file server | This project: Nginx handles SSL, rate limiting, proxies `/api/*` to FastAPI, serves React build directly |

---

## 10. Git

| Topic | One Line | Real Example |
|-------|----------|--------------|
| **git stash** | Temporarily save uncommitted changes without committing | Mid-feature, urgent hotfix needed. `git stash` → fix bug → `git stash pop` to restore your work |
| **git rebase** | Move your commits on top of another branch — linear history | Feature branch has 3 commits. `git rebase main` → your 3 commits sit on top of latest main, no merge commit |
| **git cherry-pick** | Apply one specific commit from another branch | Bug fixed on `feature/x` in commit `abc123`. `git cherry-pick abc123` applies that fix to main |
| **git reset** | Move HEAD backwards — undo commits (soft keeps changes, hard discards them) | `git reset --soft HEAD~1` → undo last commit, keep changes staged. `git reset --hard HEAD~1` → undo + discard |
| **git revert** | Create a new commit that undoes a previous commit — safe for shared branches | `git revert abc123` → new commit that reverses abc123. History preserved — safe for main branch |
| **git bisect** | Binary search through commits to find which one introduced a bug | Tests pass on v1.0, fail on v1.5. `git bisect` checks the middle commit → narrows down to exact bad commit |
| **git reset vs git revert** | reset rewrites history (unsafe on shared branches). revert adds a new undo commit (safe) | Never `git reset` on main — it rewrites shared history. Always `git revert` on main |
| **git stash vs commit** | Stash is temporary and local. Commit is permanent and shareable | Use stash for "I'll be back in 10 minutes." Use WIP commit for "I'm done for the day" |

---

## 11. Classic Design Problems

| Problem | Core Answer |
|---------|-------------|
| **URL Shortener** | Hash long URL → 6-char base62 code. Store in DB. Redirect: lookup code → 301 to long URL. Cache hot URLs in Redis |
| **Web Crawler** | BFS from seed URLs. Queue of URLs to visit. Download page → extract links → add to queue. Deduplicate with Bloom filter. Respect robots.txt |
| **Rate Limiter Design** | Token bucket in Redis. Key = IP address. `INCR` counter, set TTL. Exceeds limit → 429. Distributed: Redis atomic ops ensure consistency |
| **Design a Chat System** | WebSocket connections to chat servers. Messages to Kafka. Fan-out to recipient connections. Offline users → push notification. History in Cassandra |

---

## Topics You Should Also Know (Not in Your List)

| Topic | Why It Matters |
|-------|---------------|
| **Saga Pattern** | Distributed transactions across microservices — each step has a compensating rollback |
| **CQRS** | Separate read model from write model — reads and writes scale independently |
| **Backpressure** | Consumer tells producer to slow down when overwhelmed — prevents queue overflow |
| **Sidecar Pattern** | Deploy helper container alongside main container — Fluent Bit as log sidecar in this project |
| **mTLS** | Both client AND server verify certificates — used in service mesh for zero-trust |
| **Connection Pooling** | Reuse DB connections instead of opening new ones per request — this project uses SQLAlchemy pool |
| **Graceful Shutdown** | Finish in-flight requests before process exits — Gunicorn SIGTERM handling |
| **Observability: 3 Pillars** | Logs + Metrics + Traces — this project has all 3 (structlog, Prometheus, OTel) |

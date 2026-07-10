#!/usr/bin/env python3
"""
Production load test — 1000 requests.

Breakdown:
  700  unique valid requests      → cache misses, ONNX inference, stored in DB
  200  duplicate requests         → cache hits (20 payloads × 10 repeats)
  100  invalid room_type          → 422 validation errors, NOT stored
  ─────────────────────────────────────────────────────────────────────
  1000 total /predict calls

  + 15 DLQ entries injected directly via Redis
    (simulates what happens on real 5xx inference failures in production)

Pass/fail criteria:
  ✓ store rows == 900  (700 unique miss + 200 duplicate hit, both logged)
  ✓ cache hits  >= 180 (duplicates — ~10% tolerance for race)
  ✓ 200 count   == 900
  ✓ 422 count   == 100
  ✓ 5xx count   == 0
  ✓ DLQ size    == 15  (the injected ones)
  ✓ p95 latency < 200 ms
"""

import asyncio
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

BASE_URL     = "http://localhost:8001"
CONCURRENCY  = 30   # max in-flight requests

_API_KEY = os.getenv("NYC_API_KEY", "")
_HEADERS = {"X-API-Key": _API_KEY} if _API_KEY else {}

BOROUGHS = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
VALID_ROOM_TYPES = ["Entire home/apt", "Private room", "Shared room", "Hotel room"]
BOROUGH_COORDS = {
    "Manhattan":     (40.7831, -73.9712),
    "Brooklyn":      (40.6782, -73.9442),
    "Queens":        (40.7282, -73.7949),
    "Bronx":         (40.8448, -73.8648),
    "Staten Island": (40.5795, -74.1502),
}

# ── payload builders ──────────────────────────────────────────────────────────

def make_valid_payload(seed: int) -> dict:
    rng = random.Random(seed)
    borough = rng.choice(BOROUGHS)
    lat, lon = BOROUGH_COORDS[borough]
    return {
        "accommodates":         rng.randint(1, 8),
        "bedrooms":             float(rng.choice([0, 1, 1, 2, 2, 3, 4])),
        "bathrooms":            rng.choice([1.0, 1.0, 1.5, 2.0, 2.5]),
        "is_private_bath":      rng.choice([True, True, False]),
        "room_type":            rng.choice(VALID_ROOM_TYPES),
        "borough":              borough,
        "neighbourhood":        "",
        "latitude":             round(lat + rng.uniform(-0.04, 0.04), 4),
        "longitude":            round(lon + rng.uniform(-0.04, 0.04), 4),
        "minimum_nights":       rng.randint(1, 30),
        "host_is_superhost":    rng.choice([True, False]),
        "host_listings_count":  rng.randint(1, 15),
        "number_of_reviews":    rng.randint(0, 250),
        "number_of_reviews_ltm": rng.randint(0, 24),
        "reviews_per_month":    round(rng.uniform(0, 5), 2),
        "review_scores_rating": round(rng.uniform(3.5, 5.0), 1),
        "amenity_count":        rng.randint(5, 65),
        "has_gym":              rng.choice([True, False]),
        "has_elevator":         rng.choice([True, False]),
        "has_dryer":            rng.choice([True, False]),
        "has_air_conditioning": rng.choice([True, False]),
        "has_washer":           rng.choice([True, False]),
        "has_pool":             rng.random() < 0.1,
    }


def make_invalid_payload(seed: int) -> dict:
    p = make_valid_payload(seed)
    p["room_type"] = "Penthouse Suite"   # passes Pydantic (str), fails predictor _validate → 422
    return p


# ── DLQ injection (simulate production 5xx errors) ───────────────────────────

def inject_dlq_entries(n: int = 15) -> int:
    try:
        import redis as redis_lib
        r = redis_lib.Redis(host="localhost", port=6379, decode_responses=True)
        r.ping()
        for i in range(n):
            rng = random.Random(9000 + i)
            borough = rng.choice(BOROUGHS)
            entry = {
                "timestamp":   datetime.now(timezone.utc).isoformat(),
                "request_id":  f"sim-dlq-{i:04d}",
                "endpoint":    "/predict",
                "status_code": 500,
                "error":       rng.choice([
                    "ONNX Runtime: node compute failed",
                    "Feature engineering: unexpected NaN in dist_from_midtown",
                    "Scaler transform: shape mismatch (58,) vs (57,)",
                    "CoreML inference timeout after 30s",
                    "model file corrupted: CRC mismatch",
                ]),
                "payload": make_valid_payload(9000 + i),
            }
            r.lpush("nyc:dlq", json.dumps(entry))
        return n
    except Exception as exc:
        print(f"  ⚠  Redis not available for DLQ injection: {exc}")
        return 0


# ── async request runner ──────────────────────────────────────────────────────

async def send_one(
    client: httpx.AsyncClient,
    sem:    asyncio.Semaphore,
    idx:    int,
    payload: dict,
    label:  str,
) -> dict:
    async with sem:
        t0 = time.perf_counter()
        try:
            r = await client.post("/predict", json=payload, headers=_HEADERS, timeout=15.0)
            latency = (time.perf_counter() - t0) * 1000
            body = r.json()
            return {
                "idx":       idx,
                "label":     label,
                "status":    r.status_code,
                "latency_ms": round(latency, 2),
                "price":     body.get("price_str"),
                "cache_hit": body.get("cache_hit"),
                "request_id": body.get("request_id", "")[:8],
                "error":     body.get("detail") if r.status_code >= 400 else None,
            }
        except Exception as exc:
            latency = (time.perf_counter() - t0) * 1000
            return {"idx": idx, "label": label, "status": 0, "latency_ms": round(latency, 2), "error": str(exc)}


# ── main ─────────────────────────────────────────────────────────────────────

async def main():
    print("=" * 65)
    print("  NYC Airbnb API — Production Load Test")
    print("=" * 65)

    # ── preflight ─────────────────────────────────────────────────────────
    print("\n[0/5] Preflight check...")
    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        try:
            r = await client.get("/health", timeout=5.0)
            h = r.json()
            print(f"  API  : {h['status']}  backend={h['inference_backend']}")
            print(f"  Store: {h['store_rows']} rows before test")
            print(f"  DLQ  : {h['dlq_size']} entries before test")
            store_rows_before = h["store_rows"]
            dlq_before        = h["dlq_size"] or 0
        except Exception as exc:
            print(f"  ✗ API not reachable: {exc}")
            print("    Start the API first: PYTHONPATH=. uvicorn src.serving.api:app --port 8001")
            sys.exit(1)

    # ── DLQ injection ─────────────────────────────────────────────────────
    print("\n[1/5] Injecting 15 simulated DLQ entries (prod 5xx errors)...")
    injected = inject_dlq_entries(15)
    print(f"  Injected: {injected} entries")

    # ── build request list ────────────────────────────────────────────────
    print("\n[2/5] Building 1000 requests...")
    requests_list = []

    # 700 unique valid (seeds 0–699) → cache miss
    for seed in range(700):
        requests_list.append(("miss", make_valid_payload(seed)))

    # 200 duplicates (seeds 0–19, each 10×) → cache hit (after first warmup)
    for seed in range(20):
        for _ in range(10):
            requests_list.append(("hit", make_valid_payload(seed)))

    # 100 invalid room_type → 422
    for seed in range(100):
        requests_list.append(("invalid", make_invalid_payload(seed + 5000)))

    # Shuffle so hits/misses/errors are interleaved
    random.Random(42).shuffle(requests_list)
    assert len(requests_list) == 1000

    valid_count   = sum(1 for l, _ in requests_list if l in ("miss", "hit"))
    invalid_count = sum(1 for l, _ in requests_list if l == "invalid")
    print(f"  Valid:   {valid_count}  (700 unique miss + 200 duplicate hit)")
    print(f"  Invalid: {invalid_count}  (bad room_type → 422)")
    print(f"  Total:   {len(requests_list)}")

    # ── send requests ─────────────────────────────────────────────────────
    print(f"\n[3/5] Sending 1000 requests (concurrency={CONCURRENCY})...")
    sem     = asyncio.Semaphore(CONCURRENCY)
    results = []

    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        t_start = time.perf_counter()
        tasks = [
            send_one(client, sem, i, payload, label)
            for i, (label, payload) in enumerate(requests_list)
        ]
        results = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - t_start

    # ── analyse ───────────────────────────────────────────────────────────
    print(f"\n[4/5] Analysing results ({elapsed:.1f}s total)...")

    by_status = {}
    for r in results:
        by_status.setdefault(r["status"], []).append(r)

    latencies  = [r["latency_ms"] for r in results if r["status"] != 0]
    cache_hits = [r for r in results if r.get("cache_hit") is True]
    errors_5xx = [r for r in results if r["status"] >= 500]
    errors_0   = [r for r in results if r["status"] == 0]

    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)
    p50 = latencies_sorted[int(n * 0.50)] if n else 0
    p95 = latencies_sorted[int(n * 0.95)] if n else 0
    p99 = latencies_sorted[int(n * 0.99)] if n else 0
    avg = sum(latencies) / max(n, 1)

    # ── final check endpoints ─────────────────────────────────────────────
    async with httpx.AsyncClient(base_url=BASE_URL, headers=_HEADERS) as client:
        h_after = (await client.get("/health")).json()
        cache_s = (await client.get("/cache/stats")).json()
        dlq_r   = (await client.get("/dlq?n=5")).json()
        td_r    = (await client.get("/training-data/stats")).json()

    store_rows_after = h_after["store_rows"]
    new_rows = store_rows_after - store_rows_before

    # ── report ────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  RESULTS")
    print("=" * 65)

    print(f"\n  Throughput : {len(results) / elapsed:.0f} req/s over {elapsed:.1f}s")
    print(f"  Total sent : {len(results)}")
    print()

    status_codes = sorted(by_status.keys())
    for code in status_codes:
        count = len(by_status[code])
        bar   = "█" * (count // 10)
        print(f"  HTTP {code}  : {count:>5}  {bar}")

    print()
    print(f"  Cache hits : {len(cache_hits)}")
    print(f"  Cache rate : {cache_s['hit_rate_pct']:.1f}% ({cache_s['hits']} hits / {cache_s['hits'] + cache_s['misses']} total)")
    print()
    print(f"  Latency (ms)")
    print(f"    avg : {avg:.1f}")
    print(f"    p50 : {p50:.1f}")
    print(f"    p95 : {p95:.1f}")
    print(f"    p99 : {p99:.1f}")
    print(f"    min : {latencies_sorted[0]:.1f}" if latencies_sorted else "")
    print(f"    max : {latencies_sorted[-1]:.1f}" if latencies_sorted else "")
    print()
    print(f"  DB store")
    print(f"    Rows before test : {store_rows_before}")
    print(f"    Rows after test  : {store_rows_after}")
    print(f"    New rows added   : {new_rows}  (expect 900 = 700 miss + 200 hit)")
    print(f"    Avg price stored : ${td_r['avg_price']}")
    print()
    print(f"  DLQ")
    print(f"    Entries before: {dlq_before}")
    print(f"    Total entries : {dlq_r['size']}  (expect {dlq_before + injected} = {dlq_before} before + {injected} injected + any real 5xx)")
    print(f"    5xx from test : {len(errors_5xx)}")
    print()
    if dlq_r["entries"]:
        print("  DLQ sample (last 5 entries):")
        for e in dlq_r["entries"][:5]:
            ts  = e["timestamp"][:19]
            err = e["error"][:55]
            print(f"    [{ts}] {e['status_code']} {err}")

    # ── pass/fail ─────────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  PASS / FAIL")
    print("─" * 65)

    checks = [
        ("HTTP 200 count == 900",     len(by_status.get(200, [])) == 900),
        ("HTTP 422 count == 100",     len(by_status.get(422, [])) == 100),
        ("HTTP 5xx count == 0",       len(errors_5xx) == 0),
        ("Connection errors == 0",    len(errors_0) == 0),
        ("New DB rows == 900",        new_rows == 900),
        ("Cache hits >= 180",         len(cache_hits) >= 180),
        ("DLQ size == before + 15",   dlq_r["size"] == dlq_before + injected),
        ("p95 latency < 200 ms",      p95 < 200),
    ]

    passed = 0
    for label, ok in checks:
        icon = "✓" if ok else "✗"
        print(f"  {icon}  {label}")
        if ok:
            passed += 1

    print("─" * 65)
    if passed == len(checks):
        print(f"  ALL {passed}/{len(checks)} CHECKS PASSED — ready for deployment")
    else:
        print(f"  {passed}/{len(checks)} PASSED — {len(checks) - passed} failed (see above)")
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(main())

/**
 * k6 smoke + load test — NYC Airbnb Price Prediction API
 *
 * What k6 is:
 *   A Go-based load testing tool. You write test scripts in JavaScript,
 *   k6 executes them with N virtual users (VUs) running in parallel.
 *   Each VU loops through the default() function continuously.
 *
 * What this script tests:
 *   1. /health        — lightweight liveness check (should be very fast)
 *   2. /predict       — the hot path: auth + cache + ONNX inference + routing
 *   3. /predict-batch — 3-listing batch (tests concurrent write to SQLite store)
 *   4. /predict (bad) — intentionally wrong field to exercise validation path
 *
 * Thresholds (CI fails if any breach):
 *   - p95 latency on /predict < 500 ms
 *   - p95 latency on /health  < 50 ms
 *   - Overall error rate       < 1%
 *   - check() pass rate        > 99%
 *
 * Key concepts to understand:
 *   VU (Virtual User)  — one concurrent "user" running the loop
 *   stages             — ramp shape: controls VU count over time
 *   check()            — assertion inside a VU; failures counted but don't stop the test
 *   thresholds         — pass/fail gate evaluated at the END of the run
 *   tags / { name }    — label requests so thresholds can be per-endpoint
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate }          from 'k6/metrics';

// ── custom metric: tracks non-2xx responses across all requests ────────────────
const errorRate = new Rate('custom_error_rate');

// ── test configuration ────────────────────────────────────────────────────────
export const options = {
  /**
   * stages: ramp shape
   *   10s  0 → 20 VUs   — warm-up: lets the OS + app settle
   *   30s  20 VUs steady — the actual load window we measure
   *   10s  20 → 0 VUs   — graceful ramp-down (in-flight requests finish)
   *
   * Learning note:
   *   A flat ramp-up catches "cold start" bugs. A sudden spike would use
   *   { duration: '0s', target: 20 } then hold — useful for circuit-breaker tests.
   */
  stages: [
    { duration: '10s', target: 20 },
    { duration: '30s', target: 20 },
    { duration: '10s', target: 0  },
  ],

  /**
   * thresholds: CI gates
   *   Keys are metric names. Tag filters ({name:...}) scope them per-request.
   *   The CI step fails if ANY threshold is breached.
   */
  thresholds: {
    // Per-endpoint latency (tagged by name: param in requests below)
    'http_req_duration{name:predict}':       ['p(95)<500'],
    'http_req_duration{name:predict_cache}': ['p(95)<50'],   // cache hits should be <50ms
    'http_req_duration{name:health}':        ['p(95)<50'],

    // Overall error rate across all requests
    'http_req_failed':   ['rate<0.01'],   // < 1% HTTP errors
    'custom_error_rate': ['rate<0.01'],   // < 1% non-2xx (more strict: includes 4xx)

    // All check() assertions must pass ≥99% of the time
    'checks': ['rate>0.99'],
  },
};

// ── fixtures ──────────────────────────────────────────────────────────────────
const BASE_URL = __ENV.K6_API_URL || 'http://localhost:8001';
const API_KEY  = __ENV.K6_API_KEY  || '';

const HEADERS = {
  'Content-Type': 'application/json',
  'X-API-Key':    API_KEY,
};

// Valid Manhattan listing — hits the model, exercises feature engineering
const LISTING_MANHATTAN = JSON.stringify({
  accommodates:               2,
  bedrooms:                   1.0,
  bathrooms:                  1.0,
  is_private_bath:            true,
  room_type:                  'Entire home/apt',
  borough:                    'Manhattan',
  neighbourhood:              "Hell's Kitchen",
  latitude:                   40.7638,
  longitude:                  -73.9918,
  minimum_nights:             2,
  host_is_superhost:          false,
  host_listings_count:        1,
  number_of_reviews:          30,
  number_of_reviews_ltm:      10,
  reviews_per_month:          1.2,
  review_scores_rating:       4.8,
  review_scores_accuracy:     4.9,
  review_scores_cleanliness:  4.7,
  review_scores_checkin:      4.9,
  review_scores_communication:5.0,
  review_scores_location:     4.9,
  review_scores_value:        4.6,
  amenity_count:              30,
  has_gym:                    false,
  has_elevator:               true,
  has_dryer:                  true,
  has_air_conditioning:       true,
  has_washer:                 true,
  has_pool:                   false,
});

// Brooklyn listing — different borough, exercises neighbourhood lookup path
const LISTING_BROOKLYN = JSON.stringify({
  accommodates: 3, bedrooms: 1.0, bathrooms: 1.0,
  is_private_bath: true, room_type: 'Private room',
  borough: 'Brooklyn', neighbourhood: 'Williamsburg',
  latitude: 40.7143, longitude: -73.9551,
  minimum_nights: 3, host_is_superhost: true, host_listings_count: 2,
  number_of_reviews: 80, number_of_reviews_ltm: 20, reviews_per_month: 2.5,
  review_scores_rating: 4.6, review_scores_accuracy: 4.7,
  review_scores_cleanliness: 4.8, review_scores_checkin: 4.9,
  review_scores_communication: 5.0, review_scores_location: 4.8,
  review_scores_value: 4.5, amenity_count: 22,
  has_gym: false, has_elevator: false, has_dryer: true,
  has_air_conditioning: true, has_washer: true, has_pool: false,
});

// Batch payload: 3 listings → exercises SQLite concurrent write
const BATCH_PAYLOAD = JSON.stringify([
  JSON.parse(LISTING_MANHATTAN),
  JSON.parse(LISTING_BROOKLYN),
  { accommodates: 2, bedrooms: 1.0, bathrooms: 1.0,
    is_private_bath: true, room_type: 'Entire home/apt',
    borough: 'Queens', neighbourhood: 'Astoria',
    latitude: 40.7722, longitude: -73.9301,
    minimum_nights: 2, host_is_superhost: false, host_listings_count: 1,
    number_of_reviews: 30, number_of_reviews_ltm: 10, reviews_per_month: 1.2,
    review_scores_rating: 4.8, review_scores_accuracy: 4.9,
    review_scores_cleanliness: 4.7, review_scores_checkin: 4.9,
    review_scores_communication: 5.0, review_scores_location: 4.9,
    review_scores_value: 4.6, amenity_count: 30,
    has_gym: false, has_elevator: true, has_dryer: true,
    has_air_conditioning: true, has_washer: true, has_pool: false },
]);

// ── VU loop ──────────────────────────────────────────────────────────────────
/**
 * default() is the function each VU calls in a loop for the test duration.
 * Think of it as: one user sending requests repeatedly.
 *
 * Learning note on ordering:
 *   We make the same /predict call twice in a row. The second call should
 *   almost always hit the Redis cache (same payload → same cache key).
 *   If cache_hit latency is NOT lower than the first call, something is wrong
 *   with the caching layer — the threshold on predict_cache would catch this.
 */
export default function () {

  // ── 1. Health check ─────────────────────────────────────────────────────────
  const health = http.get(`${BASE_URL}/health/live`, {
    headers: HEADERS,
    tags:    { name: 'health' },
  });
  check(health, {
    'health 200':    (r) => r.status === 200,
    'health fast':   (r) => r.timings.duration < 100,
  });
  errorRate.add(health.status !== 200);

  // ── 2. Predict (cache miss — first request for this payload) ─────────────────
  const predict = http.post(`${BASE_URL}/predict`, LISTING_MANHATTAN, {
    headers: HEADERS,
    tags:    { name: 'predict' },
  });
  check(predict, {
    'predict 200':         (r) => r.status === 200,
    'predict has price':   (r) => JSON.parse(r.body).price_usd > 0,
    'predict has model':   (r) => JSON.parse(r.body).model !== undefined,
    'predict has req_id':  (r) => JSON.parse(r.body).request_id !== undefined,
  });
  errorRate.add(predict.status !== 200);

  // ── 3. Predict again (should be a cache hit — same payload) ──────────────────
  const cached = http.post(`${BASE_URL}/predict`, LISTING_MANHATTAN, {
    headers: HEADERS,
    tags:    { name: 'predict_cache' },
  });
  check(cached, {
    'cache 200':      (r) => r.status === 200,
    'cache hit flag': (r) => JSON.parse(r.body).cache_hit === true,
  });
  errorRate.add(cached.status !== 200);

  // ── 4. Brooklyn listing (different borough → different neighbourhood path) ────
  const brooklyn = http.post(`${BASE_URL}/predict`, LISTING_BROOKLYN, {
    headers: HEADERS,
    tags:    { name: 'predict' },
  });
  check(brooklyn, { 'brooklyn 200': (r) => r.status === 200 });
  errorRate.add(brooklyn.status !== 200);

  // ── 5. Batch (3 listings → concurrent SQLite writes from one request) ─────────
  const batch = http.post(`${BASE_URL}/predict-batch`, BATCH_PAYLOAD, {
    headers: HEADERS,
    tags:    { name: 'predict_batch' },
  });
  check(batch, {
    'batch 200':            (r) => r.status === 200,
    'batch has 3 results':  (r) => JSON.parse(r.body).predictions?.length === 3,
  });
  errorRate.add(batch.status !== 200);

  /**
   * sleep(1): pause 1 second between iterations per VU.
   *
   * Learning note:
   *   Without sleep, each VU hammers as fast as possible → unrealistic.
   *   With sleep(1) and 20 VUs → ~20 req/s steady state, which matches
   *   our rate limit of 60/minute (20 req/s × 3 endpoints = 60 req/s total).
   *   Remove the sleep to simulate a traffic spike and test rate limiting.
   */
  sleep(1);
}

/**
 * handleSummary: runs once after the test completes.
 * Writes a human-readable + machine-readable summary to disk so
 * CI can upload it as an artifact.
 */
export function handleSummary(data) {
  // Print to stdout so CI logs show key numbers immediately
  console.log('\n── Load Test Summary ──────────────────────────────────────');
  const dur  = data.metrics.http_req_duration;
  const reqs = data.metrics.http_reqs;
  if (dur && reqs) {
    console.log(`  Total requests : ${reqs.values.count}`);
    console.log(`  Req/s          : ${reqs.values.rate.toFixed(1)}`);
    console.log(`  p50 latency    : ${dur.values['p(50)'].toFixed(1)} ms`);
    console.log(`  p95 latency    : ${dur.values['p(95)'].toFixed(1)} ms`);
    console.log(`  p99 latency    : ${dur.values['p(99)'].toFixed(1)} ms`);
  }
  console.log('────────────────────────────────────────────────────────────\n');

  return {
    'load-tests/results/summary.json': JSON.stringify(data, null, 2),
    stdout: '',   // suppress default k6 summary (we printed our own above)
  };
}
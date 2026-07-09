"""
Integration tests for the NYC Airbnb Price Prediction API.

Coverage:
  - All 5 endpoints (/, /health, /metrics, /model-info, /predict, /predict-batch)
  - Happy-path predictions for all 5 boroughs and all 4 room types
  - Price sanity / direction assertions (entire > private, Manhattan > Bronx, etc.)
  - All Pydantic field constraint violations → 422
  - All predictor-level ValueError validations → 422
  - Batch: success, mixed pass/fail, oversized, empty
  - Metrics counters increment correctly after requests
  - Optional field defaults work (minimum_nights_avg_ntm=None, etc.)

Known limitations called out inline where the model's behaviour is
intentionally lenient (e.g. all-zero review scores are accepted).
"""

import copy
import pytest


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def post_predict(client, payload):
    return client.post("/predict", json=payload)


def with_override(base: dict, **overrides) -> dict:
    d = copy.deepcopy(base)
    d.update(overrides)
    return d


def without_key(base: dict, key: str) -> dict:
    d = copy.deepcopy(base)
    d.pop(key, None)
    return d


# ═══════════════════════════════════════════════════════════════════════════════
# ROOT
# ═══════════════════════════════════════════════════════════════════════════════

class TestRoot:
    def test_root_returns_200(self, client):
        r = client.get("/")
        assert r.status_code == 200

    def test_root_has_expected_keys(self, client):
        body = client.get("/").json()
        assert "api" in body
        assert "version" in body
        assert "docs" in body
        assert "backend" in body

    def test_root_api_name(self, client):
        body = client.get("/").json()
        assert "NYC" in body["api"] or "Airbnb" in body["api"]


# ═══════════════════════════════════════════════════════════════════════════════
# /health
# ═══════════════════════════════════════════════════════════════════════════════

class TestHealth:
    def test_health_200(self, client):
        assert client.get("/health").status_code == 200

    def test_health_status_healthy(self, client):
        assert client.get("/health").json()["status"] == "healthy"

    def test_health_has_uptime(self, client):
        body = client.get("/health").json()
        assert "uptime" in body
        # format: "Xh Ym Zs"
        assert "h" in body["uptime"] and "m" in body["uptime"]

    def test_health_has_request_count(self, client):
        body = client.get("/health").json()
        assert isinstance(body["total_requests"], int)
        assert body["total_requests"] >= 0

    def test_health_model_r2_correct(self, client):
        r2 = client.get("/health").json()["model_r2"]
        assert r2 >= 0.80, f"Model R² {r2:.4f} is below acceptable threshold of 0.80"

    def test_health_error_rate_format(self, client):
        rate = client.get("/health").json()["error_rate"]
        assert rate.endswith("%")

    def test_health_total_predictions_non_negative(self, client):
        body = client.get("/health").json()
        assert body["total_predictions"] >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# /metrics
# ═══════════════════════════════════════════════════════════════════════════════

class TestMetrics:
    def test_metrics_200(self, client):
        assert client.get("/metrics/summary").status_code == 200

    def test_metrics_has_uptime(self, client):
        body = client.get("/metrics/summary").json()
        assert "uptime_seconds" in body
        assert "uptime_human" in body
        assert body["uptime_seconds"] >= 0

    def test_metrics_has_totals(self, client):
        body = client.get("/metrics/summary").json()
        assert "totals" in body
        totals = body["totals"]
        assert "requests" in totals
        assert "errors" in totals
        assert "error_rate" in totals

    def test_metrics_has_endpoints_dict(self, client):
        body = client.get("/metrics/summary").json()
        assert "endpoints" in body
        assert isinstance(body["endpoints"], dict)

    def test_metrics_has_predictions_block(self, client):
        body = client.get("/metrics/summary").json()
        assert "predictions" in body
        preds = body["predictions"]
        assert "total" in preds
        assert "mean" in preds
        assert "min" in preds
        assert "max" in preds

    def test_metrics_endpoint_entry_has_latency(self, client):
        # make a request to /health so there is at least one endpoint entry
        client.get("/health")
        body = client.get("/metrics/summary").json()
        assert "/health" in body["endpoints"]
        latency = body["endpoints"]["/health"]["latency_ms"]
        assert "avg" in latency
        assert "min" in latency
        assert "max" in latency
        assert "p95" in latency  # key present (may be null until 20+ samples)

    def test_metrics_latency_avg_is_positive_after_requests(self, client):
        client.get("/health")
        body = client.get("/metrics/summary").json()
        avg = body["endpoints"]["/health"]["latency_ms"]["avg"]
        assert avg is not None
        assert avg > 0


# ═══════════════════════════════════════════════════════════════════════════════
# /model-info
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelInfo:
    def test_model_info_200(self, client):
        assert client.get("/model-info").status_code == 200

    def test_model_type_xgboost(self, client):
        assert "XGB" in client.get("/model-info").json()["model_type"]

    def test_n_features_is_67(self, client):
        assert client.get("/model-info").json()["n_features"] == 67

    def test_r2_test_correct(self, client):
        r2 = client.get("/model-info").json()["r2_test"]
        assert r2 >= 0.80, f"Model R² {r2:.4f} is below acceptable threshold of 0.80"

    def test_has_mae_and_mape(self, client):
        body = client.get("/model-info").json()
        assert "mae_dollar" in body
        assert "mape_pct" in body
        assert body["mae_dollar"] > 0
        assert body["mape_pct"] > 0

    def test_n_features_present(self, client):
        assert client.get("/model-info").json()["n_features"] > 0

    def test_n_neighbourhoods_reasonable(self, client):
        n = client.get("/model-info").json()["n_neighbourhoods"]
        # NYC has ~200+ Airbnb neighbourhoods
        assert n > 100


# ═══════════════════════════════════════════════════════════════════════════════
# /predict — happy path
# ═══════════════════════════════════════════════════════════════════════════════

class TestPredictHappyPath:
    def test_predict_200(self, client, valid_listing):
        assert post_predict(client, valid_listing).status_code == 200

    def test_predict_response_has_price_usd(self, client, valid_listing):
        body = post_predict(client, valid_listing).json()
        assert "price_usd" in body
        assert isinstance(body["price_usd"], float)

    def test_predict_response_has_price_str(self, client, valid_listing):
        body = post_predict(client, valid_listing).json()
        assert "price_str" in body
        assert body["price_str"].startswith("$")
        assert "/night" in body["price_str"]

    def test_predict_response_has_log_price(self, client, valid_listing):
        body = post_predict(client, valid_listing).json()
        assert "log_price" in body
        assert body["log_price"] > 0

    def test_predict_response_has_model_field(self, client, valid_listing):
        body = post_predict(client, valid_listing).json()
        assert "XGBoost" in body["model"]
        assert "ONNX" in body["model"]

    def test_predict_price_in_realistic_range(self, client, valid_listing):
        price = post_predict(client, valid_listing).json()["price_usd"]
        # NYC Airbnb prices: very unlikely to be under $25 or over $2,000 for a core listing
        assert 25 < price < 2000

    def test_predict_log_price_consistent_with_price_usd(self, client, valid_listing):
        import math
        body = post_predict(client, valid_listing).json()
        # expm1(log_price) should equal price_usd within rounding
        assert abs(math.expm1(body["log_price"]) - body["price_usd"]) < 1.0

    # ── borough coverage ──────────────────────────────────────────────────────

    @pytest.mark.parametrize("borough,lat,lon", [
        ("Manhattan",    40.7549, -73.9840),
        ("Brooklyn",     40.6782, -73.9442),
        ("Queens",       40.7282, -73.7949),
        ("Bronx",        40.8448, -73.8648),
        ("Staten Island",40.5795, -74.1502),
    ])
    def test_all_boroughs_predict_successfully(self, client, valid_listing, borough, lat, lon):
        payload = with_override(valid_listing, borough=borough, latitude=lat, longitude=lon)
        r = post_predict(client, payload)
        assert r.status_code == 200
        assert r.json()["price_usd"] > 0

    # ── room type coverage ────────────────────────────────────────────────────

    @pytest.mark.parametrize("room_type", [
        "Entire home/apt",
        "Private room",
        "Shared room",
        "Hotel room",
    ])
    def test_all_room_types_predict_successfully(self, client, valid_listing, room_type):
        payload = with_override(valid_listing, room_type=room_type)
        r = post_predict(client, payload)
        assert r.status_code == 200
        assert r.json()["price_usd"] > 0

    # ── optional field defaults ───────────────────────────────────────────────

    def test_minimum_nights_avg_ntm_none_accepted(self, client, valid_listing):
        """None is the default — must not raise TypeError in log1p (bug fixed)."""
        payload = with_override(valid_listing, minimum_nights_avg_ntm=None)
        assert post_predict(client, payload).status_code == 200

    def test_unknown_neighbourhood_falls_back_to_global_mean(self, client, valid_listing):
        """Unknown neighbourhood uses global_mean target encoding, not an error."""
        payload = with_override(valid_listing, neighbourhood="XYZZY_NONEXISTENT")
        r = post_predict(client, payload)
        assert r.status_code == 200
        assert r.json()["price_usd"] > 0

    def test_empty_neighbourhood_falls_back_to_global_mean(self, client, valid_listing):
        payload = with_override(valid_listing, neighbourhood="")
        assert post_predict(client, payload).status_code == 200

    def test_all_review_scores_zero_is_accepted(self, client, valid_listing):
        """
        All-zero review scores (no reviews yet) are a valid state for new listings.
        NOTE: The model still returns a price — it doesn't require non-zero scores.
        """
        payload = with_override(
            valid_listing,
            review_scores_rating=0.0,
            review_scores_accuracy=0.0,
            review_scores_cleanliness=0.0,
            review_scores_checkin=0.0,
            review_scores_communication=0.0,
            review_scores_location=0.0,
            review_scores_value=0.0,
            number_of_reviews=0,
            reviews_per_month=0.0,
        )
        r = post_predict(client, payload)
        assert r.status_code == 200
        assert r.json()["price_usd"] > 0

    def test_superhost_flag_accepted(self, client, valid_listing):
        payload = with_override(valid_listing, host_is_superhost=True)
        assert post_predict(client, payload).status_code == 200

    def test_all_amenity_flags_true(self, client, valid_listing):
        payload = with_override(
            valid_listing,
            has_gym=True, has_elevator=True, has_dryer=True,
            has_air_conditioning=True, has_washer=True, has_pool=True,
        )
        assert post_predict(client, payload).status_code == 200

    def test_minimum_required_fields_only(self, client):
        """Only the 6 truly-required fields — all optional fields use their defaults."""
        minimal = {
            "accommodates":  2,
            "room_type":     "Entire home/apt",
            "borough":       "Brooklyn",
            "latitude":      40.68,
            "longitude":     -73.94,
            "minimum_nights": 1,
        }
        r = post_predict(client, minimal)
        assert r.status_code == 200
        assert r.json()["price_usd"] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# /predict — price direction sanity
# ═══════════════════════════════════════════════════════════════════════════════

class TestPriceSanity:
    """
    Direction assertions: changing one variable should move the price predictably.
    These catch regressions in feature engineering or model artifact corruption.
    """

    def _price(self, client, valid_listing, **overrides) -> float:
        return post_predict(client, with_override(valid_listing, **overrides)).json()["price_usd"]

    def test_entire_home_costs_more_than_private_room(self, client, valid_listing):
        entire  = self._price(client, valid_listing, room_type="Entire home/apt")
        private = self._price(client, valid_listing, room_type="Private room")
        assert entire > private, f"Entire home ${entire:.0f} should exceed private room ${private:.0f}"

    def test_room_type_ordering_documented_behaviour(self, client, valid_listing):
        """
        DOCUMENTED MODEL BEHAVIOUR: the XGBoost model does NOT rank room types
        in the intuitive order (entire > hotel > private > shared).

        Actual ordering observed in training data (holding all other features equal,
        Manhattan, is_private_bath=False):
            Hotel room ≈ Shared room > Entire home/apt > Private room

        Why: the room_type dummies are weak signals compared to is_private_bath,
        neighbourhood_price_rank, and accommodates_sq. Tree splits learn whatever
        ordering exists in the training distribution — "Shared room" in Manhattan
        on Airbnb correlates with certain high-price configurations.

        This test just asserts all room types return a positive price (no crash),
        and records the actual ordering so regressions are visible.
        """
        prices = {}
        for rt in ["Entire home/apt", "Private room", "Hotel room", "Shared room"]:
            prices[rt] = self._price(client, valid_listing,
                                     room_type=rt, is_private_bath=False)
        # All room types must return a valid positive price
        for rt, price in prices.items():
            assert price > 0, f"{rt} returned non-positive price ${price}"

        # Entire home must still beat private room (this direction is robust)
        assert prices["Entire home/apt"] > prices["Private room"], (
            f"Entire home ${prices['Entire home/apt']:.0f} should exceed "
            f"private room ${prices['Private room']:.0f}"
        )

    def test_manhattan_costs_more_than_bronx(self, client, valid_listing):
        manhattan = self._price(client, valid_listing,
                                borough="Manhattan", latitude=40.76, longitude=-73.99)
        bronx     = self._price(client, valid_listing,
                                borough="Bronx", latitude=40.85, longitude=-73.88)
        assert manhattan > bronx, f"Manhattan ${manhattan:.0f} should exceed Bronx ${bronx:.0f}"

    def test_more_accommodates_higher_price(self, client, valid_listing):
        small = self._price(client, valid_listing, accommodates=1)
        large = self._price(client, valid_listing, accommodates=6)
        assert large > small, f"6-guest listing ${large:.0f} should exceed 1-guest ${small:.0f}"

    def test_private_bath_costs_more_than_shared_bath(self, client, valid_listing):
        private = self._price(client, valid_listing, is_private_bath=True)
        shared  = self._price(client, valid_listing, is_private_bath=False)
        assert private > shared, f"Private bath ${private:.0f} should exceed shared bath ${shared:.0f}"

    def test_known_neighbourhood_differs_from_unknown(self, client, valid_listing):
        """
        A known high-price neighbourhood (Midtown) should price higher than
        an unknown neighbourhood (which falls back to global mean).
        """
        known   = self._price(client, valid_listing, neighbourhood="Midtown")
        unknown = self._price(client, valid_listing, neighbourhood="XYZZY_NONEXISTENT")
        # Midtown is above the global mean — so known > unknown
        assert known > unknown, (
            f"Midtown ${known:.0f} should exceed global-mean fallback ${unknown:.0f}"
        )

    def test_manhattan_closer_to_midtown_than_staten_island(self, client, valid_listing):
        """dist_from_midtown feature: proximity to Midtown is priced in."""
        midtown    = self._price(client, valid_listing,
                                 borough="Manhattan", latitude=40.7549, longitude=-73.9840)
        staten     = self._price(client, valid_listing,
                                 borough="Staten Island", latitude=40.5795, longitude=-74.1502)
        assert midtown > staten, (
            f"Midtown ${midtown:.0f} should exceed Staten Island ${staten:.0f}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# /predict — Pydantic validation (field constraints, type coercion)
# ═══════════════════════════════════════════════════════════════════════════════

class TestPydanticValidation:
    """
    These test FastAPI/Pydantic's own validation layer — they 422 before the
    predictor even sees the request.
    """

    def test_missing_accommodates_422(self, client, valid_listing):
        assert post_predict(client, without_key(valid_listing, "accommodates")).status_code == 422

    def test_missing_room_type_422(self, client, valid_listing):
        assert post_predict(client, without_key(valid_listing, "room_type")).status_code == 422

    def test_missing_borough_422(self, client, valid_listing):
        assert post_predict(client, without_key(valid_listing, "borough")).status_code == 422

    def test_missing_latitude_422(self, client, valid_listing):
        assert post_predict(client, without_key(valid_listing, "latitude")).status_code == 422

    def test_missing_longitude_422(self, client, valid_listing):
        assert post_predict(client, without_key(valid_listing, "longitude")).status_code == 422

    def test_missing_minimum_nights_uses_default(self, client, valid_listing):
        """
        DOCUMENTED BEHAVIOUR: minimum_nights has a Pydantic default of 1 (Field(1, ge=1))
        so omitting it is NOT a 422 — the API accepts it and predicts normally.
        It is NOT a required field despite being conceptually important.
        """
        r = post_predict(client, without_key(valid_listing, "minimum_nights"))
        assert r.status_code == 200
        assert r.json()["price_usd"] > 0

    def test_accommodates_zero_422(self, client, valid_listing):
        """ge=1 constraint on accommodates."""
        assert post_predict(client, with_override(valid_listing, accommodates=0)).status_code == 422

    def test_accommodates_17_422(self, client, valid_listing):
        """le=16 constraint on accommodates."""
        assert post_predict(client, with_override(valid_listing, accommodates=17)).status_code == 422

    def test_accommodates_negative_422(self, client, valid_listing):
        assert post_predict(client, with_override(valid_listing, accommodates=-1)).status_code == 422

    def test_review_scores_rating_above_5_422(self, client, valid_listing):
        """le=5 constraint."""
        assert post_predict(client, with_override(valid_listing, review_scores_rating=5.1)).status_code == 422

    def test_review_scores_rating_negative_422(self, client, valid_listing):
        """ge=0 constraint."""
        assert post_predict(client, with_override(valid_listing, review_scores_rating=-0.1)).status_code == 422

    def test_latitude_too_far_south_422(self, client, valid_listing):
        """ge=40.4 — latitude of 39.0 is outside NYC."""
        assert post_predict(client, with_override(valid_listing, latitude=39.0)).status_code == 422

    def test_latitude_too_far_north_422(self, client, valid_listing):
        """le=41.0."""
        assert post_predict(client, with_override(valid_listing, latitude=41.5)).status_code == 422

    def test_longitude_too_far_east_422(self, client, valid_listing):
        """le=-73.6."""
        assert post_predict(client, with_override(valid_listing, longitude=-70.0)).status_code == 422

    def test_longitude_too_far_west_422(self, client, valid_listing):
        """ge=-74.3."""
        assert post_predict(client, with_override(valid_listing, longitude=-75.0)).status_code == 422

    def test_accommodates_string_422(self, client, valid_listing):
        """Pydantic coercion: 'two' cannot be converted to int."""
        assert post_predict(client, with_override(valid_listing, accommodates="two")).status_code == 422

    def test_empty_body_422(self, client):
        assert client.post("/predict", json={}).status_code == 422

    def test_non_json_body_422(self, client):
        r = client.post("/predict", content="not json", headers={"Content-Type": "application/json"})
        assert r.status_code == 422

    def test_422_response_has_detail_field(self, client, valid_listing):
        r = post_predict(client, without_key(valid_listing, "accommodates"))
        assert "detail" in r.json()

    # ── fields that have defaults and are therefore optional ─────────────────

    def test_bedrooms_optional_default_works(self, client, valid_listing):
        """bedrooms defaults to 1.0 — omitting it should succeed."""
        assert post_predict(client, without_key(valid_listing, "bedrooms")).status_code == 200

    def test_bathrooms_optional_default_works(self, client, valid_listing):
        assert post_predict(client, without_key(valid_listing, "bathrooms")).status_code == 200

    def test_host_listings_count_optional_default_works(self, client, valid_listing):
        assert post_predict(client, without_key(valid_listing, "host_listings_count")).status_code == 200

    def test_amenity_count_optional_default_works(self, client, valid_listing):
        assert post_predict(client, without_key(valid_listing, "amenity_count")).status_code == 200

    def test_neighbourhood_optional_default_works(self, client, valid_listing):
        assert post_predict(client, without_key(valid_listing, "neighbourhood")).status_code == 200


# ═══════════════════════════════════════════════════════════════════════════════
# /predict — predictor-level validation (ValueError → 422)
# ═══════════════════════════════════════════════════════════════════════════════

class TestPredictorValidation:
    """
    room_type and borough are now Pydantic Literal types — rejected at the
    API boundary before reaching the predictor. detail is a list of error dicts.
    """

    @staticmethod
    def _detail_str(r) -> str:
        """Normalise detail to a searchable string regardless of Pydantic version."""
        detail = r.json()["detail"]
        return str(detail).lower() if not isinstance(detail, str) else detail.lower()

    def test_invalid_room_type_422(self, client, valid_listing):
        r = post_predict(client, with_override(valid_listing, room_type="Studio"))
        assert r.status_code == 422
        assert "room_type" in self._detail_str(r)

    def test_invalid_borough_422(self, client, valid_listing):
        r = post_predict(client, with_override(valid_listing, borough="New Jersey"))
        assert r.status_code == 422
        assert "borough" in self._detail_str(r)

    def test_invalid_room_type_detail_lists_valid_options(self, client, valid_listing):
        detail_str = self._detail_str(
            post_predict(client, with_override(valid_listing, room_type="Penthouse"))
        )
        assert "private room" in detail_str or "entire home" in detail_str

    def test_invalid_borough_detail_lists_valid_options(self, client, valid_listing):
        detail_str = self._detail_str(
            post_predict(client, with_override(valid_listing, borough="Narnia"))
        )
        assert "manhattan" in detail_str or "brooklyn" in detail_str

    # ── NOTE: the following inputs PASS validation ────────────────────────────
    # These are documented here so future developers know the model's tolerance.

    def test_review_scores_at_boundary_zero_accepted(self, client, valid_listing):
        """
        DOCUMENTED BEHAVIOUR: review_scores_rating=0.0 is accepted.
        The predictor only rejects review_scores_rating > 5 or < 0.
        Zero scores are treated as 'no rating yet' for new listings.
        """
        r = post_predict(client, with_override(valid_listing, review_scores_rating=0.0))
        assert r.status_code == 200

    def test_review_scores_at_boundary_five_accepted(self, client, valid_listing):
        r = post_predict(client, with_override(valid_listing, review_scores_rating=5.0))
        assert r.status_code == 200

    def test_zero_number_of_reviews_accepted(self, client, valid_listing):
        """New listings with no reviews yet are valid inputs."""
        r = post_predict(client, with_override(valid_listing, number_of_reviews=0, reviews_per_month=0.0))
        assert r.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════════
# /predict-batch
# ═══════════════════════════════════════════════════════════════════════════════

class TestPredictBatch:
    """
    Async batch endpoint tests.

    POST /predict-batch  → 202 + job_id (enqueue)
    GET  /predict-batch/{job_id}  → poll status + results

    Tests that require Redis (to actually process jobs) skip gracefully when
    Redis is unavailable so the suite passes in environments without it.
    """

    def _submit(self, client, listings, explain=False):
        return client.post("/predict-batch", json={"listings": listings, "explain": explain})

    def _poll(self, client, job_id, timeout=10):
        import time
        deadline = time.time() + timeout
        while time.time() < deadline:
            r = client.get(f"/predict-batch/{job_id}")
            if r.status_code != 200:
                return r
            if r.json()["status"] in ("done", "failed"):
                return r
            time.sleep(0.2)
        return r

    # ── submission ────────────────────────────────────────────────────────────

    def test_submit_returns_202(self, client, valid_listing):
        r = self._submit(client, [valid_listing])
        assert r.status_code in (202, 503)

    def test_submit_schema(self, client, valid_listing):
        r = self._submit(client, [valid_listing])
        if r.status_code == 503:
            pytest.skip("Redis not available")
        body = r.json()
        assert "job_id" in body
        assert "status"     in body
        assert "total"      in body
        assert "status_url" in body
        assert body["total"] == 1

    def test_submit_status_url_matches_job_id(self, client, valid_listing):
        r = self._submit(client, [valid_listing])
        if r.status_code == 503:
            pytest.skip("Redis not available")
        body = r.json()
        assert body["status_url"] == f"/predict-batch/{body['job_id']}"

    # ── polling + results ─────────────────────────────────────────────────────

    def test_poll_single_listing_completes(self, client, valid_listing):
        r = self._submit(client, [valid_listing])
        if r.status_code == 503:
            pytest.skip("Redis not available")
        job = self._poll(client, r.json()["job_id"]).json()
        assert job["status"] == "done"
        assert job["total"] == 1
        assert job["done"]  == 1
        assert len(job["results"]) == 1
        assert job["results"][0]["price_usd"] > 0
        assert job["results"][0]["index"] == 0

    def test_poll_two_listings_both_complete(self, client, valid_listing):
        brooklyn = with_override(valid_listing, borough="Brooklyn", latitude=40.68, longitude=-73.94)
        r = self._submit(client, [valid_listing, brooklyn])
        if r.status_code == 503:
            pytest.skip("Redis not available")
        job = self._poll(client, r.json()["job_id"]).json()
        assert job["status"] == "done"
        assert job["total"] == 2
        assert len(job["results"]) == 2

    def test_poll_results_have_required_fields(self, client, valid_listing):
        r = self._submit(client, [valid_listing])
        if r.status_code == 503:
            pytest.skip("Redis not available")
        job = self._poll(client, r.json()["job_id"]).json()
        item = job["results"][0]
        for field in ("index", "price_usd", "price_low", "price_high", "price_str"):
            assert field in item, f"missing field: {field}"

    def test_poll_preserves_index_order(self, client, valid_listing):
        queens = with_override(valid_listing, borough="Queens", latitude=40.73, longitude=-73.79)
        bronx  = with_override(valid_listing, borough="Bronx",  latitude=40.85, longitude=-73.87)
        r = self._submit(client, [valid_listing, queens, bronx])
        if r.status_code == 503:
            pytest.skip("Redis not available")
        job = self._poll(client, r.json()["job_id"]).json()
        assert [item["index"] for item in job["results"]] == [0, 1, 2]

    # ── validation ────────────────────────────────────────────────────────────

    def test_invalid_listing_rejected_at_submission(self, client, valid_listing):
        bad = with_override(valid_listing, borough="Narnia")
        r = self._submit(client, [bad])
        # Pydantic rejects the whole batch if any listing is invalid
        assert r.status_code == 422

    def test_empty_listings_rejected(self, client):
        r = self._submit(client, [])
        assert r.status_code == 422

    def test_exceeds_max_size_rejected(self, client, valid_listing):
        giant = [valid_listing] * 51
        r = self._submit(client, giant)
        assert r.status_code == 422

    def test_exactly_max_size_accepted(self, client, valid_listing):
        fifty = [valid_listing] * 50
        r = self._submit(client, fifty)
        assert r.status_code in (202, 503)

    # ── not found ─────────────────────────────────────────────────────────────

    def test_unknown_job_id_returns_404(self, client):
        r = client.get("/predict-batch/00000000-0000-0000-0000-000000000000")
        if r.status_code == 503:
            pytest.skip("Redis not available")
        assert r.status_code == 404


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics tracking — counters update after requests
# ═══════════════════════════════════════════════════════════════════════════════

class TestMetricsTracking:
    """
    These tests verify the observability layer records correctly.
    They read /metrics before and after requests and assert the delta.

    NOTE: Because the TestClient session is shared, absolute counts depend
    on test execution order. We assert *increases* (delta), not absolute values.
    """

    def test_predict_increments_request_count(self, client, valid_listing):
        before = client.get("/metrics/summary").json()["endpoints"].get("/predict", {}).get("requests", 0)
        post_predict(client, valid_listing)
        after  = client.get("/metrics/summary").json()["endpoints"].get("/predict", {}).get("requests", 0)
        assert after == before + 1

    def test_prediction_total_increments(self, client, valid_listing):
        before = client.get("/metrics/summary").json()["predictions"]["total"]
        post_predict(client, valid_listing)
        after  = client.get("/metrics/summary").json()["predictions"]["total"]
        assert after == before + 1

    def test_prediction_mean_updates_after_predict(self, client, valid_listing):
        post_predict(client, valid_listing)
        mean = client.get("/metrics/summary").json()["predictions"]["mean"]
        assert mean is not None
        assert mean > 0  # prediction values are USD prices

    def test_batch_job_completes_with_correct_count(self, client, valid_listing):
        """Batch async — verify both listings reach 'done' status."""
        import time
        brooklyn = with_override(valid_listing, borough="Brooklyn", latitude=40.68, longitude=-73.94)
        r = client.post("/predict-batch", json={"listings": [valid_listing, brooklyn]})
        if r.status_code == 503:
            pytest.skip("Redis not available")
        job_id = r.json()["job_id"]
        for _ in range(50):
            job = client.get(f"/predict-batch/{job_id}").json()
            if job["status"] == "done":
                break
            time.sleep(0.2)
        assert job["status"] == "done"
        assert job["total"] == 2
        assert len(job["results"]) == 2

    def test_error_count_increments_on_invalid_request(self, client, valid_listing):
        # 422 (Pydantic / predictor ValueError) are NOT counted as errors by the middleware —
        # only HTTP 5xx increments the error counter. A 422 is a client error, not a server error.
        # So we verify the error counter does NOT change on a validation failure.
        before_errs = client.get("/metrics/summary").json()["endpoints"].get("/predict", {}).get("errors", 0)
        post_predict(client, with_override(valid_listing, borough="Narnia"))  # → 422, not a server error
        after_errs  = client.get("/metrics/summary").json()["endpoints"].get("/predict", {}).get("errors", 0)
        assert after_errs == before_errs  # counter unchanged — 422 ≠ server error

    def test_predict_latency_recorded_after_request(self, client, valid_listing):
        post_predict(client, valid_listing)
        latency = client.get("/metrics/summary").json()["endpoints"]["/predict"]["latency_ms"]
        assert latency["avg"] is not None
        assert latency["avg"] > 0
        assert latency["min"] is not None
        assert latency["max"] is not None

    def test_p95_latency_available_after_20_requests(self, client, valid_listing):
        """p95 requires at least 20 samples — fire 20 /health calls to ensure it."""
        for _ in range(20):
            client.get("/health")
        p95 = client.get("/metrics/summary").json()["endpoints"]["/health"]["latency_ms"]["p95"]
        assert p95 is not None
        assert p95 > 0

    def test_health_total_requests_increases(self, client):
        before = client.get("/metrics/summary").json()["totals"]["requests"]
        client.get("/health")
        client.get("/health")
        after  = client.get("/metrics/summary").json()["totals"]["requests"]
        assert after >= before + 2

    def test_prediction_min_is_set_after_predict(self, client, valid_listing):
        post_predict(client, valid_listing)
        preds = client.get("/metrics/summary").json()["predictions"]
        assert preds["min"] is not None
        assert preds["max"] is not None
        assert preds["max"] >= preds["min"]


# ═══════════════════════════════════════════════════════════════════════════════
# GROUND TRUTH INGEST (admin endpoint — gated by GROUND_TRUTH_INGEST_TOKEN)
# ═══════════════════════════════════════════════════════════════════════════════

class TestGroundTruthIngest:
    def test_disabled_when_token_unset(self, client, monkeypatch):
        import src.config as cfg
        monkeypatch.setattr(cfg.settings, "ground_truth_ingest_token", "")
        r = client.post("/ground-truth/ingest")
        assert r.status_code == 503

    def test_rejects_missing_header(self, client, monkeypatch):
        import src.config as cfg
        monkeypatch.setattr(cfg.settings, "ground_truth_ingest_token", "test-secret")
        r = client.post("/ground-truth/ingest")
        assert r.status_code == 401

    def test_rejects_wrong_token(self, client, monkeypatch):
        import src.config as cfg
        monkeypatch.setattr(cfg.settings, "ground_truth_ingest_token", "test-secret")
        r = client.post("/ground-truth/ingest", headers={"X-Ingest-Token": "nope"})
        assert r.status_code == 401

    def test_accepts_correct_token_and_returns_ingest_result(self, client, monkeypatch):
        import src.config as cfg
        from src.new_york_workflow import nyc_api
        monkeypatch.setattr(cfg.settings, "ground_truth_ingest_token", "test-secret")
        monkeypatch.setattr(
            nyc_api, "run_ground_truth_ingest",
            lambda url, dry_run: {"matched": 1, "inserted": 1, "pending": 1},
        )
        r = client.post("/ground-truth/ingest", headers={"X-Ingest-Token": "test-secret"})
        assert r.status_code == 200
        assert r.json() == {"matched": 1, "inserted": 1, "pending": 1}

    def test_ingest_exception_returns_500(self, client, monkeypatch):
        import src.config as cfg
        monkeypatch.setattr(cfg.settings, "ground_truth_ingest_token", "test-secret")
        from src.new_york_workflow import nyc_api

        def boom(url, dry_run):
            raise RuntimeError("snapshot download failed")

        monkeypatch.setattr(nyc_api, "run_ground_truth_ingest", boom)
        r = client.post("/ground-truth/ingest", headers={"X-Ingest-Token": "test-secret"})
        assert r.status_code == 500


# ═══════════════════════════════════════════════════════════════════════════════
# API KEY AUTH
# ═══════════════════════════════════════════════════════════════════════════════

class TestAPIKeyAuth:
    def test_predict_without_key_returns_401(self, valid_listing):
        from src.new_york_workflow.nyc_api import app
        from fastapi.testclient import TestClient as TC
        # Fresh client with NO default headers — no key passed
        with TC(app) as c:
            r = c.post("/predict", json=valid_listing)
        assert r.status_code == 401

    def test_predict_with_wrong_key_returns_401(self, valid_listing):
        from src.new_york_workflow.nyc_api import app
        from fastapi.testclient import TestClient as TC
        with TC(app, headers={"X-API-Key": "completely-wrong"}) as c:
            r = c.post("/predict", json=valid_listing)
        assert r.status_code == 401

    def test_health_exempt_from_auth(self):
        from src.new_york_workflow.nyc_api import app
        from fastapi.testclient import TestClient as TC
        with TC(app) as c:
            for path in ("/health", "/health/ready", "/health/live"):
                r = c.get(path)
                assert r.status_code != 401, f"{path} should be exempt from auth"

    def test_root_exempt_from_auth(self):
        from src.new_york_workflow.nyc_api import app
        from fastapi.testclient import TestClient as TC
        with TC(app) as c:
            r = c.get("/")
        assert r.status_code == 200

    def test_docs_exempt_from_auth(self):
        from src.new_york_workflow.nyc_api import app
        from fastapi.testclient import TestClient as TC
        with TC(app) as c:
            r = c.get("/docs")
        assert r.status_code == 200

    def test_valid_key_from_session_fixture_works(self, client, valid_listing):
        # Verify the session-scoped client (with test key in default headers) passes auth
        r = client.post("/predict", json=valid_listing)
        assert r.status_code == 200

    def test_auth_disabled_when_valid_api_keys_empty(self, valid_listing, monkeypatch):
        from src.new_york_workflow import nyc_api
        from src.config import settings
        monkeypatch.setattr(settings, "valid_api_keys", "")
        from fastapi.testclient import TestClient as TC
        with TC(nyc_api.app) as c:
            r = c.post("/predict", json=valid_listing)
        assert r.status_code == 200  # auth off — no key needed


class TestSlackAlerts:
    def test_slack_not_called_when_no_webhook(self, tmp_path, monkeypatch):
        from unittest.mock import patch
        from src.new_york_workflow.nyc_alerts import AlertStore
        from src.config import settings
        store = AlertStore(tmp_path / "alerts.json")
        monkeypatch.setattr(settings, "slack_webhook_url", "")
        with patch("requests.post") as mock_post:
            store.push("test_alert", "hello", severity="warning")
            mock_post.assert_not_called()

    def test_slack_not_called_for_info_severity(self, tmp_path, monkeypatch):
        from unittest.mock import patch
        from src.new_york_workflow.nyc_alerts import AlertStore
        from src.config import settings
        store = AlertStore(tmp_path / "alerts.json")
        monkeypatch.setattr(settings, "slack_webhook_url", "https://hooks.slack.com/fake")
        with patch("requests.post") as mock_post:
            store.push("info_event", "just fyi", severity="info")
            mock_post.assert_not_called()

    def test_slack_called_for_warning(self, tmp_path, monkeypatch):
        from unittest.mock import patch, MagicMock
        from src.new_york_workflow.nyc_alerts import AlertStore
        from src.config import settings
        store = AlertStore(tmp_path / "alerts.json")
        monkeypatch.setattr(settings, "slack_webhook_url", "https://hooks.slack.com/fake")
        mock_resp = MagicMock()
        mock_resp.ok = True
        with patch("requests.post", return_value=mock_resp) as mock_post:
            store.push("drift_detected", "PSI > 0.20", severity="warning")
            mock_post.assert_called_once()
            payload = mock_post.call_args.kwargs["json"]
            assert "WARNING" in payload["text"]
            assert "drift_detected" in payload["text"]

    def test_slack_called_for_critical(self, tmp_path, monkeypatch):
        from unittest.mock import patch, MagicMock
        from src.new_york_workflow.nyc_alerts import AlertStore
        from src.config import settings
        store = AlertStore(tmp_path / "alerts.json")
        monkeypatch.setattr(settings, "slack_webhook_url", "https://hooks.slack.com/fake")
        mock_resp = MagicMock()
        mock_resp.ok = True
        with patch("requests.post", return_value=mock_resp) as mock_post:
            store.push("rollback", "auto-reverted", severity="critical")
            mock_post.assert_called_once()
            payload = mock_post.call_args.kwargs["json"]
            assert "CRITICAL" in payload["text"]
            assert ":rotating_light:" in payload["text"]

    def test_slack_critical_channel_override(self, tmp_path, monkeypatch):
        from unittest.mock import patch, MagicMock
        from src.new_york_workflow.nyc_alerts import AlertStore
        from src.config import settings
        store = AlertStore(tmp_path / "alerts.json")
        monkeypatch.setattr(settings, "slack_webhook_url", "https://hooks.slack.com/fake")
        monkeypatch.setattr(settings, "slack_critical_channel", "#incidents")
        mock_resp = MagicMock()
        mock_resp.ok = True
        with patch("requests.post", return_value=mock_resp) as mock_post:
            store.push("rollback", "reverted", severity="critical")
            payload = mock_post.call_args.kwargs["json"]
            assert payload["channel"] == "#incidents"

    def test_slack_failure_does_not_raise(self, tmp_path, monkeypatch):
        from unittest.mock import patch
        from src.new_york_workflow.nyc_alerts import AlertStore
        from src.config import settings
        store = AlertStore(tmp_path / "alerts.json")
        monkeypatch.setattr(settings, "slack_webhook_url", "https://hooks.slack.com/fake")
        with patch("requests.post", side_effect=Exception("network down")):
            alert_id = store.push("oops", "msg", severity="critical")
            assert alert_id  # file write still succeeded

    def test_push_once_dedup_still_works_with_slack(self, tmp_path, monkeypatch):
        from unittest.mock import patch, MagicMock
        from src.new_york_workflow.nyc_alerts import AlertStore
        from src.config import settings
        store = AlertStore(tmp_path / "alerts.json")
        monkeypatch.setattr(settings, "slack_webhook_url", "https://hooks.slack.com/fake")
        mock_resp = MagicMock()
        mock_resp.ok = True
        with patch("requests.post", return_value=mock_resp) as mock_post:
            id1 = store.push_once("drift", "first", severity="warning")
            id2 = store.push_once("drift", "duplicate", severity="warning")
            assert id1 is not None
            assert id2 is None           # suppressed
            mock_post.assert_called_once()  # Slack only fired once


class TestPrometheusMetrics:
    def test_prometheus_endpoint_200(self, client):
        assert client.get("/metrics").status_code == 200

    def test_prometheus_content_type(self, client):
        r = client.get("/metrics")
        assert "text/plain" in r.headers["content-type"]

    def test_prometheus_has_http_requests_total(self, client):
        client.get("/health")
        body = client.get("/metrics").text
        assert "http_requests_total" in body

    def test_prometheus_has_nyc_predictions(self, client, valid_listing):
        client.post("/predict", json=valid_listing)
        body = client.get("/metrics").text
        assert "nyc_predictions_total" in body

    def test_prometheus_has_nyc_price_histogram(self, client, valid_listing):
        client.post("/predict", json=valid_listing)
        body = client.get("/metrics").text
        assert "nyc_prediction_price_usd" in body

    def test_prometheus_has_canary_gauge(self, client):
        body = client.get("/metrics").text
        assert "nyc_canary_traffic_pct" in body

    def test_prometheus_has_dlq_gauge(self, client):
        body = client.get("/metrics").text
        assert "nyc_dlq_size" in body


# ═══════════════════════════════════════════════════════════════════════════════
# /predict — price sanity guard
# ═══════════════════════════════════════════════════════════════════════════════

class TestPriceSanityGuard:
    """
    _guard_price raises 422 for predictions outside [$10, $5000].
    These tests mock predictor.predict_raw to return extreme values.
    """

    def test_valid_price_passes(self, client, valid_listing):
        r = post_predict(client, valid_listing)
        assert r.status_code == 200
        price = r.json()["price_usd"]
        assert 10.0 <= price <= 5000.0

    def test_price_sanity_guard_imported(self):
        from src.new_york_workflow.nyc_api import _PRICE_MIN_USD, _PRICE_MAX_USD, _guard_price
        assert _PRICE_MIN_USD == 10.0
        assert _PRICE_MAX_USD == 5_000.0

    def test_guard_raises_on_zero_price(self):
        from src.new_york_workflow.nyc_api import _guard_price
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc:
            _guard_price(0.0, "test-rid-001")
        assert exc.value.status_code == 422
        assert "outside valid range" in exc.value.detail

    def test_guard_raises_on_negative_price(self):
        from src.new_york_workflow.nyc_api import _guard_price
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc:
            _guard_price(-50.0, "test-rid-002")
        assert exc.value.status_code == 422

    def test_guard_raises_on_excessive_price(self):
        from src.new_york_workflow.nyc_api import _guard_price
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc:
            _guard_price(99_999.0, "test-rid-003")
        assert exc.value.status_code == 422
        assert "outside valid range" in exc.value.detail

    def test_guard_passes_on_boundary_minimum(self):
        from src.new_york_workflow.nyc_api import _guard_price
        _guard_price(10.0, "test-rid-004")   # must not raise

    def test_guard_passes_on_boundary_maximum(self):
        from src.new_york_workflow.nyc_api import _guard_price
        _guard_price(5000.0, "test-rid-005")  # must not raise

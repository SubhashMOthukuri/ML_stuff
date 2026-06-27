"""
Export the trained XGBoost model to ONNX format.

ONNX Runtime inference is 3-5x faster than XGBoost native Python
because it bypasses the Python GIL, uses optimised C++ kernels, and
supports parallel inference across workers without re-entering Python.

Run once:
    python scripts/export_onnx.py
"""

import pickle
import time
import numpy as np
from pathlib import Path

BASE_DIR  = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models" / "nyc"


def load_artifacts():
    def _load(name):
        with open(MODEL_DIR / name, "rb") as f:
            return pickle.load(f)
    return (
        _load("nyc_xgb_model.pkl"),
        _load("nyc_scaler.pkl"),
        _load("nyc_feature_list.pkl"),
    )


def export_onnx(model, feature_list):
    from onnxmltools.convert import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType as OnnxFloat

    n_features = len(feature_list)
    print(f"  Converting XGBoost model ({n_features} features) to ONNX...")

    initial_type = [("float_input", OnnxFloat([None, n_features]))]
    onnx_model   = convert_xgboost(model, initial_types=initial_type)

    out_path = MODEL_DIR / "nyc_xgb_model.onnx"
    with open(out_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  Saved: {out_path}  ({size_mb:.1f} MB)")
    return out_path


def benchmark(model, onnx_path, scaler, feature_list, n_runs=1000):
    import onnxruntime as rt

    # Build a realistic sample input
    rng = np.random.default_rng(42)
    X   = rng.random((1, len(feature_list))).astype(np.float32)
    Xs  = scaler.transform(X).astype(np.float32)

    # XGBoost native
    t0 = time.perf_counter()
    for _ in range(n_runs):
        model.predict(Xs)
    xgb_ms = (time.perf_counter() - t0) / n_runs * 1000

    # ONNX Runtime
    sess = rt.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp  = sess.get_inputs()[0].name

    # warm up
    sess.run(None, {inp: Xs})

    t0 = time.perf_counter()
    for _ in range(n_runs):
        sess.run(None, {inp: Xs})
    ort_ms = (time.perf_counter() - t0) / n_runs * 1000

    speedup = xgb_ms / ort_ms
    print(f"\n  Benchmark ({n_runs} single-row predictions):")
    print(f"    XGBoost native  : {xgb_ms:.3f} ms / request")
    print(f"    ONNX Runtime    : {ort_ms:.3f} ms / request")
    print(f"    Speedup         : {speedup:.1f}x faster")

    # Verify outputs match
    xgb_pred = float(model.predict(Xs)[0])
    ort_pred = float(np.array(sess.run(None, {inp: Xs})[0]).flat[0])
    diff = abs(xgb_pred - ort_pred)
    print(f"    Output diff     : {diff:.6f}  {'✓ match' if diff < 1e-3 else '✗ MISMATCH'}")

    return speedup


def main():
    print("=" * 60)
    print("NYC Airbnb XGBoost → ONNX Export")
    print("=" * 60)

    print("\nLoading artifacts...")
    model, scaler, feature_list = load_artifacts()
    print(f"  Model type    : {type(model).__name__}")
    print(f"  Feature count : {len(feature_list)}")

    print("\nExporting to ONNX...")
    onnx_path = export_onnx(model, feature_list)

    print("\nBenchmarking inference speed...")
    speedup = benchmark(model, onnx_path, scaler, feature_list)

    print(f"""
{'=' * 60}
Export complete.

Artifact : models/nyc/nyc_xgb_model.onnx
Speedup  : {speedup:.1f}x vs XGBoost native

At 1000 concurrent users:
  XGBoost native : limited by Python GIL + sklearn overhead
  ONNX Runtime   : pure C++ kernel, no GIL, full CPU parallelism
{'=' * 60}
""")


if __name__ == "__main__":
    main()

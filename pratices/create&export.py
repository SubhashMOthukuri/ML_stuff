import numpy as np
import pickle
import logging
from pathlib import Path
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)

#step 1: generate fake data

def get_data():
    X , y = make_regression(n_samples= 1000, n_features= 10, noise = 20 , random_state= 42)
    return train_test_split(X, y, test_size=0.2, random_state=42)


#step 2: Scaler featurees

def scale(X_train, X_test):
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    return X_train_sc, X_test_sc, scaler

#step 3: Train XGBRegressor

def train(X_train_sc, y_train):
    search= XGBRegressor()
    model = search.fit(X_train_sc, y_train)
    return model

#step 5: Export ONNX
def export_onnx(model, n_features):
    from onnxmltools.convert import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType
    initial_type = [("float_input", FloatTensorType([None, n_features]))]

    onnx_model = convert_xgboost(model, initial_types= initial_type)
    with open("pratices/practice_model.onnx", "wb") as f:
              f.write(onnx_model.SerializeToString())
    return onnx_model


#step 6: Verify

def verify(model, scaler, n_features):
    import onnxruntime as rt
    data = np.random.default_rng(0).random((1, n_features)).astype(np.float32)
    data_sc = scaler.transform(data)
    xgb_pred = float(model.predict(data_sc)[0])
    sess = rt.InferenceSession("pratices/practice_model.onnx")
    inp = sess.get_inputs()[0].name
    ort_pred = float(np.array(sess.run(None, {inp: data_sc.astype(np.float32)})[0]).flat[0])
    print(f"XGBoost: {xgb_pred:.4f}")
    print(f"ONNX:    {ort_pred:.4f}")
    print(f"Diff:    {abs(xgb_pred - ort_pred):.6f}")

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_data()
    X_train_sc, X_test_sc, scaler = scale(X_train, X_test)
    model = train(X_train_sc, y_train)

    print("R sqaure: " , r2_score(y_test, model.predict(X_test_sc)))
    export_onnx(model, X_train.shape[1])
    verify(model, scaler, X_train.shape[1])
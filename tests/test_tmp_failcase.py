import unittest, warnings
import xgboost as xgb
import numpy as np
from hummingbird.ml import convert
from sklearn.metrics import accuracy_score
from onnxconverter_common.data_types import FloatTensorType
from onnxmltools.convert import convert_xgboost
import onnxruntime as ort

# Temporarily stashing this here, todo move this into test_xgboost_converter.py
class TestXGBoostConverterFailCase(unittest.TestCase):
    def test_branch_lt(self):
        warnings.filterwarnings("ignore")
        num_classes = 2
        X = np.array(np.random.rand(10000, 28), dtype=np.float32)
        y = np.random.randint(num_classes, size=10000)
        X_test = np.array(np.random.rand(10000, 28), dtype=np.float32)
        y_test = np.random.randint(num_classes, size=10000)

        model = xgb.XGBClassifier()
        model.fit(X, y)

        predict = model.predict(X_test)
        print('accuracy', accuracy_score(y_test, predict))


        initial_types = [("input", FloatTensorType([X.shape[0], X.shape[1]]))] # Define the inputs for the ONNX
        onnx_ml_model = convert_xgboost(model, initial_types=initial_types, target_opset=9)

        onnx_model = convert(onnx_ml_model, "onnx", X)

        session = ort.InferenceSession(onnx_model.model.SerializeToString())
        output_names = [session.get_outputs()[i].name for i in range(len(session.get_outputs()))]
        inputs = {session.get_inputs()[0].name: X_test}

        predict_2 = session.run(output_names, inputs)[0]
        print('accuracy after', accuracy_score(y_test, predict_2))


    if __name__ == "__main__":
        unittest.main()

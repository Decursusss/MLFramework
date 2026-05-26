from ml.MLAlgorithms.LinealRegression import train_linear, predict_linear
from ml.MLAlgorithms.LogicalRegression import train_logistic, predict_logistic
from ml.MLAlgorithms.RandomForest import train_random_forest, predict_random_forest
from ml.MLAlgorithms.XGBoost import train_xgboost, predict_xgboost
import os
from datetime import datetime
import json

class MlAlgorithmInterface:
    def __init__(self):
        self.MODEL_DIRECTORY = os.path.join(os.getcwd(), "models")
        self.AVAILABLE_MODELS = ["linear", "logistic", "random_forest", "xgboost"]

    def train_model(self, X:list, Y:list, model_type:str = "linear"):
        if model_type.lower() == "linear":
            model = train_linear(X, Y,
                                 lr=0.01,  #Learning rate (smaller mean more looking rate more efficient model)
                                 epochs=1000 #Epochs means how much data will be processed time
                                 )
        elif model_type.lower() == "logistic":
            model = train_logistic(X, Y,
                                   lr=0.01,  #Learning rate (smaller mean more looking rate more efficient model)
                                   epochs=1000 #Epochs means how much data will be processed time
                                   )
        elif model_type.lower() == "random_forest":
            model = train_random_forest(X, Y,
                                        n_trees=50, #Trees amount (how many trees there will be existed)
                                        max_depth=5, #Branch trees deepness (how much will be deep tree higher not better)
                                        min_samples=3 #Minimal points in list (optimal 3 but could be more)
                                        )
        elif model_type.lower() == "xgboost":
            model = train_xgboost(X, Y,
                                  n_trees=100, #Trees amount (how many trees there will be existed)
                                  max_depth=5, #Branch trees deepness (how much will be deep tree higher not better)
                                  lr=0.05, #Learning rate (smaller mean more looking rate more efficient model)
                                  lam=1.0 #Regulazation
                                  )
        else:
            raise ValueError(f"Model not found at {model_type}! Available options are {self.AVAILABLE_MODELS}")
        return model

    def predict_model(self, model, new_points:list, model_type:str = "linear"):
        if model_type.lower() == "linear":
            result = predict_linear(model, new_points)
        elif model_type.lower() == "logistic":
            result = predict_logistic(model, new_points)
        elif model_type.lower() == "random_forest":
            result = predict_random_forest(model, new_points)
        elif model_type.lower() == "xgboost":
            result = predict_xgboost(model, new_points)
        else:
            raise ValueError(f"Unknown model type {model_type}! Available models are {self.AVAILABLE_MODELS}")
        return result

    def save_model(self, model, norm_params, model_type:str = "linear", model_name: str = None):
        os.makedirs(self.MODEL_DIRECTORY, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        if model_name is None:
            model_name = model.get("type", model_type)

        if model_type.lower() in self.AVAILABLE_MODELS:
            payload = {
                "model": model,
                "norm_params": norm_params,
                "saved_at": timestamp,
            }

            path = f"{self.MODEL_DIRECTORY}/{model_name}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

            print(f"Model {model_type} saved at models/{model_name}.json")
            return model_name
        else:
            raise ValueError(f"Unknown model type {model_type}! Available models are {self.AVAILABLE_MODELS}")

    def load_model(self, model_name: str = None):
        path = f"{self.MODEL_DIRECTORY}/{model_name}.json"
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model not found at {path}")

        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        print(f"Model {payload.get("model", "").get("type", "")} loaded from models/{model_name}.json")
        return payload["model"], payload["norm_params"]
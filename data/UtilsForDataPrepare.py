import statistics
from datetime import datetime

#Helping functions for data preparation
def parse_datetime_features(dt_str, normalize=False):
    if not dt_str:
        return {}

    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", ""))
    except:
        return {}

    if normalize:
        return [float(dt.hour), float(dt.weekday()), float(dt.day), float(dt.month), float((dt.weekday() >= 5))]

    return {
        "hour": dt.hour, # 0–23 hours range
        "day_of_week": dt.weekday(), # 0=monday, 6=saturday
        "day_of_month": dt.day, # 1-31 days
        "month": dt.month, # 1-12 months
        "is_weekend": int(dt.weekday() >= 5), # 0 or 1 weekend indicator
    }

def rows_to_XY(rows, feature_cols, label_col, date_cols=None, cast=float, normalize=True, scaler="minmax"): #scalar type mean or minmax
    X_raw, y_raw = [], []
    feature_cols = list(feature_cols)
    for row in rows:
        row = dict(row)
        if date_cols:
            for col in date_cols:
                parsed = parse_datetime_features(row.get(col, ""))
                for k, v in parsed.items():
                    key = f"{col}_{k}"
                    row[key] = v
                    if key not in feature_cols:
                        feature_cols.append(key)
        try:
            features = [cast(row[col]) for col in feature_cols]
            label = cast(row[label_col])
            X_raw.append(features)
            y_raw.append(label)
        except:
            continue

    print(f"Loaded {len(X_raw)} points from {len(feature_cols)} features + Y labels from {len(y_raw)} points")
    if not normalize:
        return X_raw, y_raw, None

    n_cols = len(feature_cols)
    if scaler == "minmax":
        col_min = [min(row[j] for row in X_raw) for j in range(n_cols)]
        col_max = [max(row[j] for row in X_raw) for j in range(n_cols)]

        X = []
        for row in X_raw:
            normalized = []
            for j in range(n_cols):
                rng = col_max[j] - col_min[j]
                normalized.append((row[j] - col_min[j]) / rng if rng > 0 else 0.0)
            X.append(normalized)

        y_min = min(y_raw)
        y_max = max(y_raw)
        y_rng = y_max - y_min
        y = [(v - y_min) / y_rng if y_rng > 0 else 0.0 for v in y_raw]

        norm_params = {
            "X": {"min": col_min, "max": col_max, "cols": feature_cols, "scaler": "minmax"},
            "y": {"min": y_min, "max": y_max, "scaler": "minmax"},
        }
    elif scaler == "mean":
        col_mean = [statistics.mean(row[j] for row in X_raw) for j in range(n_cols)]
        col_std = [statistics.stdev(row[j] for row in X_raw) or 1.0 for j in range(n_cols)]

        X = []
        for row in X_raw:
            normalized = []
            for j in range(n_cols):
                normalized.append((row[j] - col_mean[j]) / col_std[j])
            X.append(normalized)

        y_mean = statistics.mean(y_raw)
        y_std = statistics.stdev(y_raw) or 1.0
        y = [(v - y_mean) / y_std for v in y_raw]

        norm_params = {
            "X": {"mean": col_mean, "std": col_std, "cols": feature_cols, "scaler": "mean"},
            "y": {"mean": y_mean, "std": y_std, "scaler": "mean"},
        }
    else:
        raise ValueError(f"Not found scaler: '{scaler}'. Available scaler: 'minmax', 'mean'")

    return X, y, norm_params

def train_test_split(X, y, test_size=0.2):
    split = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"train: {len(X_train)} points | test: {len(X_test)} points ~ split by {test_size*100}%")
    return X_train, X_test, y_train, y_test

def denormalize_y(value, norm_params):
    if norm_params["y"].get("scaler") == "mean":
        return value * norm_params["y"]["std"] + norm_params["y"]["mean"]
    return value * (norm_params["y"]["max"] - norm_params["y"]["min"]) + norm_params["y"]["min"]

def normalize_point(point, norm_params):
    if norm_params["X"].get("scaler") == "mean":
        mean = norm_params["X"]["mean"]
        std = norm_params["X"]["std"]
        return [(val - mean[j]) / std[j] for j, val in enumerate(point)]
    col_min = norm_params["X"]["min"]
    col_max = norm_params["X"]["max"]
    return [(val - col_min[j]) / (col_max[j] - col_min[j]) if (col_max[j] - col_min[j]) > 0 else 0.0 for j, val in enumerate(point)]
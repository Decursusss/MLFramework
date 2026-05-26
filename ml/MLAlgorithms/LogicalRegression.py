#

def train_logistic(X, y, lr=0.1, epochs=1000):
    n_features = len(X[0])
    w = [0.0] * n_features
    b = 0.0
    n = len(X)

    for _ in range(epochs):
        dw = [0.0] * n_features
        db = 0.0
        for i in range(n):
            z = sum(w[k] * X[i][k] for k in range(n_features)) + b
            pred = 1 / (1 + 2.718**(-z))
            err = pred - y[i]
            for k in range(n_features):
                dw[k] += err * X[i][k]
            db += err
        for k in range(n_features):
            w[k] -= lr * dw[k] / n
        b -= lr * db / n

    return {"w": w, "b": b, "type": "logical"}


def predict_logistic(model, X):
    w, b = model["w"], model["b"]
    n_features = len(w)
    results = []
    for xi in X:
        z = sum(w[k] * xi[k] for k in range(n_features)) + b
        prob = 1 / (1 + 2.718**(-z))
        label = 1 if prob >= 0.5 else 0
        results.append({"value": prob, "label": label})
    return results
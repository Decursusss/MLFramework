#

def train_linear(X, y, lr=0.01, epochs=1000):
    n_features = len(X[0])
    w = [0.0] * n_features
    b = 0.0
    n = len(X)

    for epoch in range(epochs):
        dw = [0.0] * n_features
        db = 0.0
        loss = 0.0

        for i in range(n):
            pred = sum(w[k] * X[i][k] for k in range(n_features)) + b
            err = pred - y[i]
            loss += err ** 2

            for k in range(n_features):
                dw[k] += err * X[i][k]
            db += err

        for k in range(n_features):
            w[k] -= lr * dw[k] / n
        b -= lr * db / n

    return {"w": w, "b": b, "type": "linear"}


def predict_linear(model, X):
    w, b = model["w"], model["b"]
    n_features = len(w)
    results = []
    for xi in X:
        value = sum(w[k] * xi[k] for k in range(n_features)) + b
        results.append({"value": value})
    return results
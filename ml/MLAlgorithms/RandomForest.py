#

import random

def mse(values):
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)

def best_split(X, y, n_features_sample):
    best_loss = float("inf")
    best_feat = None
    best_thresh = None

    n_features = len(X[0])
    feat_indices = random.sample(range(n_features), min(n_features_sample, n_features))

    for feat in feat_indices:
        values = sorted(set(row[feat] for row in X))
        thresholds = [(values[i] + values[i+1]) / 2 for i in range(len(values)-1)]

        for thresh in thresholds:
            left_y  = [y[i] for i in range(len(X)) if X[i][feat] <= thresh]
            right_y = [y[i] for i in range(len(X)) if X[i][feat] >  thresh]

            if not left_y or not right_y:
                continue

            loss = (len(left_y) * mse(left_y) + len(right_y) * mse(right_y)) / len(y)

            if loss < best_loss:
                best_loss = loss
                best_feat = feat
                best_thresh = thresh

    return best_feat, best_thresh

def build_tree(X, y, depth, max_depth, min_samples, n_features_sample):
    if depth >= max_depth or len(y) <= min_samples:
        return {"leaf": True, "value": sum(y) / len(y)}

    feat, thresh = best_split(X, y, n_features_sample)

    if feat is None:
        return {"leaf": True, "value": sum(y) / len(y)}

    left_idx = [i for i in range(len(X)) if X[i][feat] <= thresh]
    right_idx = [i for i in range(len(X)) if X[i][feat] >  thresh]

    left_X = [X[i] for i in left_idx]
    left_y = [y[i] for i in left_idx]
    right_X = [X[i] for i in right_idx]
    right_y = [y[i] for i in right_idx]

    return {
        "leaf":   False,
        "feat":   feat,
        "thresh": thresh,
        "left":   build_tree(left_X,  left_y,  depth+1, max_depth, min_samples, n_features_sample),
        "right":  build_tree(right_X, right_y, depth+1, max_depth, min_samples, n_features_sample),
    }

def predict_tree(node, xi):
    if node["leaf"]:
        return node["value"]
    if xi[node["feat"]] <= node["thresh"]:
        return predict_tree(node["left"],  xi)
    else:
        return predict_tree(node["right"], xi)

def train_random_forest(X, y, n_trees=100, max_depth=5, min_samples=5, n_features_sample=None, seed=42):
    """
        X: нормализованные признаки
        y: нормализованные целевые значения
        n_trees: количество деревьев (больше = точнее но медленнее)
        max_depth: глубина каждого дерева (больше = сложнее модель)
        min_samples: минимум точек в узле для разбивки
        n_features_sample : сколько признаков смотреть в каждом узле None = sqrt(n_features) - стандартное значение
    """
    random.seed(seed)
    n = len(X)
    n_features = len(X[0])

    if n_features_sample is None:
        n_features_sample = max(1, int(n_features ** 0.5))

    trees = []
    for t in range(n_trees):
        indices = [random.randint(0, n-1) for _ in range(n)]
        X_boot = [X[i] for i in indices]
        y_boot = [y[i] for i in indices]

        tree = build_tree(X_boot, y_boot, depth=0,
                           max_depth=max_depth,
                           min_samples=min_samples,
                           n_features_sample=n_features_sample)
        trees.append(tree)

    return {
        "trees": trees,
        "type": "random_forest",
        "n_trees": n_trees,
        "max_depth": max_depth,
    }

def predict_random_forest(model, X):
    trees = model["trees"]
    results = []
    for xi in X:
        preds = [predict_tree(tree, xi) for tree in trees]
        value = sum(preds) / len(preds)
        results.append({"value": value})
    return results
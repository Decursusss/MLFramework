#

import random

def mean(values):
    return sum(values) / len(values) if values else 0.0

def best_split_xgb(X, residuals, n_features_sample, lam=1.0):
    best_gain  = 0.0
    best_feat  = None
    best_thresh = None

    n = len(X)
    n_features = len(X[0])
    feat_indices = random.sample(range(n_features), min(n_features_sample, n_features))

    G_total = sum(residuals)

    for feat in feat_indices:
        values = sorted(set(row[feat] for row in X))
        thresholds = [(values[i] + values[i+1]) / 2 for i in range(len(values)-1)]

        for thresh in thresholds:
            left_idx = [i for i in range(n) if X[i][feat] <= thresh]
            right_idx = [i for i in range(n) if X[i][feat] >  thresh]

            if not left_idx or not right_idx:
                continue

            G_left = sum(residuals[i] for i in left_idx)
            G_right = sum(residuals[i] for i in right_idx)
            n_left = len(left_idx)
            n_right = len(right_idx)

            gain = (
                (G_left  ** 2) / (n_left  + lam) +
                (G_right ** 2) / (n_right + lam) -
                (G_total ** 2) / (n       + lam)
            )

            if gain > best_gain:
                best_gain = gain
                best_feat = feat
                best_thresh = thresh

    return best_feat, best_thresh

def leaf_value(residuals, lam=1.0):
    return -sum(residuals) / (len(residuals) + lam)

def build_xgb_tree(X, residuals, depth, max_depth, min_samples, n_features_sample, lam):
    if depth >= max_depth or len(residuals) <= min_samples:
        return {"leaf": True, "value": leaf_value(residuals, lam)}

    feat, thresh = best_split_xgb(X, residuals, n_features_sample, lam)

    if feat is None:
        return {"leaf": True, "value": leaf_value(residuals, lam)}

    left_idx = [i for i in range(len(X)) if X[i][feat] <= thresh]
    right_idx = [i for i in range(len(X)) if X[i][feat] >  thresh]

    left_X = [X[i] for i in left_idx]
    right_X = [X[i] for i in right_idx]
    left_r = [residuals[i] for i in left_idx]
    right_r = [residuals[i] for i in right_idx]

    return {
        "leaf": False,
        "feat": feat,
        "thresh": thresh,
        "left": build_xgb_tree(left_X,  left_r,  depth+1, max_depth, min_samples, n_features_sample, lam),
        "right": build_xgb_tree(right_X, right_r, depth+1, max_depth, min_samples, n_features_sample, lam),
    }

def predict_tree(node, xi):
    if node["leaf"]:
        return node["value"]
    if xi[node["feat"]] <= node["thresh"]:
        return predict_tree(node["left"],  xi)
    else:
        return predict_tree(node["right"], xi)

def train_xgboost(X, y,
                  n_trees=100,
                  max_depth=4,
                  lr=0.1,
                  min_samples=3,
                  lam=1.0,
                  n_features_sample=None,
                  seed=42):
    """
    X: нормализованные признаки
    y: нормализованные целевые значения
    n_trees: количество деревьев (больше = точнее)
    max_depth: глубина каждого дерева (обычно 3-6 для XGBoost)
    lr: learning rate - насколько сильно каждое дерево влияет маленький lr + много деревьев = лучше но медленнее
    lam: λ регуляризация - штраф за сложность, уменьшает переобучение
    n_features_sample: признаков на узел, None = sqrt(n_features)
    """
    random.seed(seed)
    n = len(X)
    n_features = len(X[0])

    if n_features_sample is None:
        n_features_sample = max(1, int(n_features ** 0.5))

    base_pred = mean(y)
    preds = [base_pred] * n
    trees = []
    losses = []

    for t in range(n_trees):
        residuals = [preds[i] - y[i] for i in range(n)]

        tree = build_xgb_tree(
            X, residuals,
            depth=0,
            max_depth=max_depth,
            min_samples=min_samples,
            n_features_sample=n_features_sample,
            lam=lam,
        )

        for i in range(n):
            preds[i] += lr * predict_tree(tree, X[i])

        trees.append(tree)

        loss = sum((preds[i] - y[i]) ** 2 for i in range(n)) / n
        losses.append(loss)

    return {
        "trees": trees,
        "base_pred": base_pred,
        "lr": lr,
        "type": "xgboost",
        "n_trees": n_trees,
        "max_depth": max_depth,
        "losses": losses,
    }


def predict_xgboost(model, X):
    trees = model["trees"]
    base_pred = model["base_pred"]
    lr = model["lr"]

    results = []
    for xi in X:
        value = base_pred
        for tree in trees:
            value += lr * predict_tree(tree, xi)
        results.append({"value": value})

    return results
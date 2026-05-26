def mae(y_true, y_pred):
    return sum(abs(y_true[i] - y_pred[i]) for i in range(len(y_true))) / len(y_true)

def rmse(y_true, y_pred):
    return (sum((y_true[i] - y_pred[i])**2 for i in range(len(y_true))) / len(y_true)) ** 0.5

def mape(y_true, y_pred):
    return sum(abs((y_true[i] - y_pred[i]) / y_true[i]) for i in range(len(y_true))) / len(y_true) * 100

def evaluate(model_name, y_true, y_pred, norm_params, denorm_fn):
    y_true_real = [denorm_fn(v, norm_params) for v in y_true]
    y_pred_real = [denorm_fn(v, norm_params) for v in y_pred]

    print(f"\nModel Type: {model_name}")
    print(f"MAE:  {mae(y_true_real, y_pred_real):.0f}   ← average value error")
    print(f"RMSE: {rmse(y_true_real, y_pred_real):.0f}   ← squared value error")
    print(f"MAPE: {mape(y_true_real, y_pred_real):.1f}%  ← average % error\n")
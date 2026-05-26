import matplotlib.pyplot as plt

def plot_predictions(y_true, y_pred, norm_params, denorm_fn, title="Predictions"):
    y_true_real = [denorm_fn(v, norm_params) for v in y_true]
    y_pred_real = [denorm_fn(v, norm_params) for v in y_pred]

    plt.figure(figsize=(12, 4))
    plt.plot(y_true_real, label="Real Price", color="blue")
    plt.plot(y_pred_real, label="Predicted Price",  color="red", alpha=0.7)
    plt.title(title)
    plt.legend()
    plt.savefig(f"evaluation/reports/{title}.png")
    plt.show()

def plot_errors(y_true, y_pred, norm_params, denorm_fn, title="Errors"):
    y_true_real = [denorm_fn(v, norm_params) for v in y_true]
    y_pred_real = [denorm_fn(v, norm_params) for v in y_pred]
    errors = [y_true_real[i] - y_pred_real[i] for i in range(len(y_true_real))]

    plt.figure(figsize=(12, 4))
    plt.bar(range(len(errors)), errors, color=["red" if e < 0 else "green" for e in errors])
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title(title)
    plt.savefig(f"evaluation/reports/{title}.png")
    plt.show()
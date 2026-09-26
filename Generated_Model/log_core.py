import csv
from pathlib import Path
from datetime import datetime

_cols = [
    "timestamp",
    # hiperparametry architektury
    "latent_dim", "depth", "width_first",
    # hiperparametry treningu
    "batch_size", "learning_rate", "l2", "dropout", "beta(KL)",
    # metadane treningu
    "epochs", "stopped_epoch", "params", "train_time_sec",
    # metryki rekonstrukcji / anomalii
    "train_mse", "val_mse", "anom_mse", "threshold", "roc_auc", "pr_auc",
    # mnetryki w mm
    "train_Rmse", "val_Rmse", "anom_Rmse",
    # metryki kl-divergence
    "kl_divergence", "train_kl", "val_kl", "latent_var_std"
]

def log_run(path: str = "runs.csv", **data):
    """
    """
    p = Path(path)
    # jeśli plik nie istnieje → nagłówek
    if not p.exists():
        with p.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=_cols).writeheader()

    row = {c: data.get(c, "") for c in _cols}
    row["timestamp"] = datetime.now().isoformat(timespec="seconds")

    with p.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=_cols).writerow(row)

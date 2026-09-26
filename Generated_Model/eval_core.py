import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def evaluate(model, X_train, X_test, X_anom, perc=95):
    r_tr = model.predict(X_train, verbose=0)
    r_te = model.predict(X_test , verbose=0)
    r_an = model.predict(X_anom , verbose=0)

    mse_tr = np.mean((X_train - r_tr)**2, axis=1)
    mse_te = np.mean((X_test  - r_te)**2, axis=1)
    mse_an = np.mean((X_anom  - r_an)**2, axis=1)

    thr = np.percentile(mse_tr, perc)
    y_t  = np.concatenate([np.zeros_like(mse_te), np.ones_like(mse_an)])
    y_s  = np.concatenate([mse_te, mse_an])

    return dict(
        train_mse = float(np.mean(mse_tr)),
        val_mse   = float(np.mean(mse_te)),
        threshold = float(thr),
        roc_auc   = float(roc_auc_score(y_t, y_s)),
        pr_auc    = float(average_precision_score(y_t, y_s))
    )

import pandas as pd
import numpy as np
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sklearn.inspection import permutation_importance
from .config import FIXED_N_FEATURES


def select_features_nested(X_train_df, y_train, random_state=42):
    feature_names = X_train_df.columns.tolist()
    X_train_arr = X_train_df.values

    # 1. Correlation Filter
    corr_matrix = pd.DataFrame(X_train_arr, columns=feature_names).corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    kept_features = [f for f in feature_names if f not in to_drop]

    X_train_filtered = X_train_df[kept_features]
    X_train_arr = X_train_filtered.values
    current_feat_names = X_train_filtered.columns.tolist()

    # 2. LASSO
    try:
        lasso = CoxnetSurvivalAnalysis(
            l1_ratio=1.0, alpha_min_ratio=0.01, n_alphas=100, fit_baseline_model=False
        )
        lasso.fit(X_train_arr, y_train)
        coef_max = np.abs(lasso.coef_).max(axis=1)
        nonzero_idx = np.where(coef_max > 0)[0]
        if len(nonzero_idx) == 0:
            screened_idx = np.arange(X_train_arr.shape[1])
        else:
            screened_idx = (
                np.argsort(coef_max)[-100:] if len(nonzero_idx) > 100 else nonzero_idx
            )
    except:
        screened_idx = np.arange(X_train_arr.shape[1])

    X_subset = X_train_arr[:, screened_idx]

    # 3. Random Survival Forest Permutation Importance
    rsf = RandomSurvivalForest(
        n_estimators=200,
        min_samples_split=10,
        min_samples_leaf=15,
        max_features="sqrt",
        n_jobs=-1,
        random_state=random_state,
    )
    rsf.fit(X_subset, y_train)
    result = permutation_importance(
        rsf, X_subset, y_train, n_repeats=5, random_state=random_state, n_jobs=-1
    )

    top_k_idx = screened_idx[np.argsort(result.importances_mean)[::-1]][
        :FIXED_N_FEATURES
    ]
    return [current_feat_names[i] for i in top_k_idx]

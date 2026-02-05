import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.impute import SimpleImputer
from sksurv.metrics import (
    concordance_index_ipcw,
    concordance_index_censored,
    cumulative_dynamic_auc,
)
from sksurv.util import Surv
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.compare import compare_survival

from src.config import *
from src.utils import seed_everything
from src.data_loader import load_data
from src.preprocessing import select_features_nested
from src.models import DeepRMoE


def plot_expert_weights(model, feature_names, fold_idx, output_dir):
    w0 = model.expert0.linear.weight.detach().cpu().numpy().flatten()
    w1 = model.expert1.linear.weight.detach().cpu().numpy().flatten()
    df = pd.DataFrame({"Feature": feature_names, "Expert A": w0, "Expert B": w1})
    df["Magnitude"] = df["Expert A"].abs() + df["Expert B"].abs()
    df = df.sort_values(by="Magnitude", ascending=False)
    df_melt = df.melt(
        id_vars=["Feature"],
        value_vars=["Expert A", "Expert B"],
        var_name="Expert",
        value_name="Weight",
    )
    plt.figure(figsize=(12, 10))
    sns.set_style("whitegrid")
    palette = {"Expert A": "#1f77b4", "Expert B": "#d62728"}
    sns.barplot(
        data=df_melt, y="Feature", x="Weight", hue="Expert", palette=palette, orient="h"
    )
    plt.axvline(x=0, color="black", linewidth=0.8, linestyle="--")
    plt.title(
        f"Expert Weight Comparison (Fold {fold_idx})", fontsize=16, fontweight="bold"
    )
    plt.xlabel("Weight Coefficient", fontsize=14)
    plt.ylabel("Features", fontsize=14)
    plt.legend(title=None, loc="lower right", fontsize=12)
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"Expert_Weights_Fold_{fold_idx}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_time_dependent_auc(auc_results):
    years = []
    means = []
    stds = []
    for t in sorted(auc_results.keys()):
        vals = auc_results[t]
        if len(vals) > 0:
            years.append(t / 365)
            means.append(np.mean(vals))
            stds.append(np.std(vals))
    if not years:
        return
    plt.figure(figsize=(10, 6))
    plt.plot(years, means, marker="o", color="#8e44ad", linewidth=2, label="Hard MoE")
    plt.fill_between(
        years,
        np.array(means) - np.array(stds),
        np.array(means) + np.array(stds),
        color="#8e44ad",
        alpha=0.1,
    )
    for x, y in zip(years, means):
        plt.text(x, y + 0.015, f"{y:.3f}", ha="center", fontweight="bold", fontsize=11)
    plt.title("Time-dependent AUC (Hard Inference)", fontsize=15, fontweight="bold")
    plt.xlabel("Time (Years)", fontsize=12)
    plt.ylabel("AUC", fontsize=12)
    plt.ylim(0.5, 1.0)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "HardMoE_AUC.png"), dpi=300)
    plt.close()


def run_evaluation():
    seed_everything(RANDOM_SEED)

    X_df, T, E = load_data()
    y_struct = Surv.from_arrays(event=E.astype(bool), time=T)

    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_SEED)
    c_index_scores = []
    p_value_scores = []
    auc_results = {t: [] for t in range(365, 1826 + 1, 365)}

    print(f"   Output Directory: {OUTPUT_DIR}")

    for i, (train_idx, val_idx) in tqdm(enumerate(rkf.split(X_df, E)), total=25):
        try:
            X_train_raw, X_val_raw = X_df.iloc[train_idx], X_df.iloc[val_idx]
            y_train, y_val = y_struct[train_idx], y_struct[val_idx]

            # Preprocessing
            imputer = SimpleImputer(strategy="mean")
            pt = PowerTransformer(method="yeo-johnson", standardize=True)

            X_train_imp = imputer.fit_transform(X_train_raw)
            X_train_pt = pt.fit_transform(X_train_imp)
            X_val_imp = imputer.transform(X_val_raw)
            X_val_pt = pt.transform(X_val_imp)

            X_train_df = pd.DataFrame(
                X_train_pt, columns=X_df.columns, index=X_train_raw.index
            )
            X_val_df = pd.DataFrame(
                X_val_pt, columns=X_df.columns, index=X_val_raw.index
            )

            # Feature Selection
            selected_feats = select_features_nested(
                X_train_df, y_train, random_state=RANDOM_SEED + i
            )
            X_tr_sel = X_train_df[selected_feats].values
            X_val_sel = X_val_df[selected_feats].values

            # Model Training
            model = DeepRMoE(
                in_features=len(selected_feats),
                lr=LEARNING_RATE,
                epochs=MAX_EPOCHS,
                patience=PATIENCE,
            )
            model.fit(X_tr_sel, y_train, val_data=(X_val_sel, y_val))

            # Prediction
            pred, expert_indices = model.predict(X_val_sel, return_experts=True)

            # --- Evaluation: Risk Score Median Split (p-value, KM Curve) ---
            cutoff = np.median(pred)
            mask_high = pred >= cutoff
            group_indicator = np.zeros(len(y_val), dtype=int)
            group_indicator[mask_high] = 1

            try:
                chisq, p_val_fold = compare_survival(
                    y_val, group_indicator, return_stats=False
                )
                p_value_scores.append(p_val_fold)
            except Exception as e:
                p_val_fold = 1.0

            # --- Visualization (First 5 folds only) ---
            if (i + 1) in [1, 2, 3, 4, 5]:
                # 1. Expert Weights
                plot_expert_weights(model, selected_feats, i + 1, OUTPUT_DIR)

                # 2. KM Curve (Median Split)
                time_low, prob_low = kaplan_meier_estimator(
                    y_val["event"][~mask_high], y_val["time"][~mask_high]
                )
                time_high, prob_high = kaplan_meier_estimator(
                    y_val["event"][mask_high], y_val["time"][mask_high]
                )

                plt.figure(figsize=(8, 6))
                plt.step(
                    time_low,
                    prob_low,
                    where="post",
                    label="Low Risk (Median Split)",
                    color="green",
                )
                plt.step(
                    time_high,
                    prob_high,
                    where="post",
                    label="High Risk (Median Split)",
                    color="red",
                )
                plt.ylim(0, 1.05)
                plt.title(
                    f"Kaplan-Meier Curve (Fold {i+1})\nLog-rank p={p_val_fold:.4f}"
                )
                plt.xlabel("Time (Days)")
                plt.ylabel("Survival Probability")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(
                    os.path.join(OUTPUT_DIR, f"KM_Curve_Fold_{i+1}.png"), dpi=300
                )
                plt.close()

            # C-Index
            try:
                c_idx = concordance_index_ipcw(y_train, y_val, pred)[0]
            except:
                c_idx = concordance_index_censored(y_val["event"], y_val["time"], pred)[
                    0
                ]
            c_index_scores.append(c_idx)

            # Time-dependent AUC
            times = np.array(list(auc_results.keys()))
            max_train = y_train["time"].max()
            min_val = y_val["time"].min()
            max_val = y_val["time"].max()
            safe_max = min(max_val, max_train - 1e-5)
            valid_times = times[(times > min_val) & (times < safe_max)]

            if len(valid_times) > 0:
                try:
                    auc_vals, _ = cumulative_dynamic_auc(
                        y_train, y_val, pred, valid_times
                    )
                    for t, auc in zip(valid_times, auc_vals):
                        auc_results[t].append(auc)
                except:
                    pass

        except Exception as e:
            print(f"Error in fold {i}: {e}")
            continue

    # Final Reporting
    if len(c_index_scores) > 0:
        mean_c = np.mean(c_index_scores)
        std_c = np.std(c_index_scores)
        mean_p = np.mean(p_value_scores)
        std_p = np.std(p_value_scores)

        print("\n" + "=" * 30)
        print(f"FINAL PERFORMANCE (Median Split based on Risk Score)")
        print(f"Mean C-Index: {mean_c:.5f} (±{std_c:.5f})")
        print(f"Mean Log-rank p-value: {mean_p:.5f} (±{std_p:.5f})")
        print("=" * 30)

        with open(os.path.join(OUTPUT_DIR, "DeepMoE_Metrics.txt"), "w") as f:
            f.write(f"Mean C-Index: {mean_c:.5f}\n")
            f.write(f"Std C-Index : {std_c:.5f}\n")
            f.write(f"Mean Log-rank p-value: {mean_p:.5f}\n")
            f.write(f"Std Log-rank p-value : {std_p:.5f}\n")

            for t in sorted(auc_results.keys()):
                vals = auc_results[t]
                if len(vals) > 0:
                    f.write(
                        f"Year {t // 365}: {np.mean(vals):.5f} ± {np.std(vals):.5f}\n"
                    )

        plot_time_dependent_auc(auc_results)


if __name__ == "__main__":
    run_evaluation()

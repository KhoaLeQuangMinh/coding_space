import os
import pandas as pd
import scipy.stats

base_dir = '/Users/khoale/Downloads/ablation_result/working/coding_space/Hope Implementation/checkpoints/'

variants = {
    'ce': 'ablation_loss_ce_fold',
    'full': 'ablation_loss_full_fold',
    'triplet_only': 'ablation_loss_triplet_only_fold'
}

metrics = ['MCI Acc', 'MCI F1', 'MCI Prec', 'MCI SEN', 'MCI AUC']

results = {k: {m: [] for m in metrics} for k in variants}

for var_key, var_folder_prefix in variants.items():
    for fold in range(1, 6):
        csv_path = os.path.join(base_dir, f"{var_folder_prefix}{fold}", "test_metrics_best_2c.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            for m in metrics:
                results[var_key][m].append(df.iloc[0][m])

print("=== Raw Arrays ===")
for var_key in variants:
    print(f"--- {var_key} ---")
    for m in metrics:
        print(f"{m}: {results[var_key][m]}")

print("\n=== P-Values (Proposed Triplet vs CE Baseline) ===")
for m in metrics:
    stat, p = scipy.stats.ttest_rel(results['triplet_only'][m], results['ce'][m])
    print(f"{m}: p = {p:.5f}")
    
print("\n=== P-Values (Proposed Triplet vs HOPE / Full) ===")
for m in metrics:
    stat, p = scipy.stats.ttest_rel(results['triplet_only'][m], results['full'][m])
    print(f"{m}: p = {p:.5f}")

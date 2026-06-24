import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import scipy.stats

data_dir = "/Users/khoale/Downloads/analysis_output_tSNE/extracted_features/"
out_dir = "/Users/khoale/Downloads/ablation_result/statistical_proof/"
os.makedirs(out_dir, exist_ok=True)

variants = {
    'ce': 'CE (Baseline)',
    'full': 'HOPE (Full)',
    'triplet_only_margin0.3': 'Proposed (Triplet Only) (Margin 0.3)',
    'triplet_only_margin3.0': 'Proposed (Triplet Only) (Margin 3.0)',
    'triplet_only_ema0.5_margin0.0': 'Proposed (Triplet Only) (EMA 0.5, Margin 0.0)'
}

colors = {'ce': 'gray', 'full': 'blue', 'triplet_only_margin0.3': 'red', 'triplet_only_margin3.0': 'purple', 'triplet_only_ema0.5_margin0.0': 'orange'}

plt.figure(figsize=(8, 8))

auc_results = {k: [] for k in variants.keys()}

# For mean ROC plotting
tprs = {k: [] for k in variants.keys()}
mean_fpr = np.linspace(0, 1, 100)

for var_key, var_name in variants.items():
    for fold in range(1, 6):
        csv_path = os.path.join(data_dir, f"{var_key}_best_2c_net_fold{fold}.csv")
        if not os.path.exists(csv_path):
            continue
            
        df = pd.read_csv(csv_path)
        
        # Filter to only sMCI and pMCI
        df_mci = df[df['True Label'].isin(['sMCI', 'pMCI'])].copy()
        
        if len(df_mci) == 0:
            continue
            
        # Target: pMCI is 1, sMCI is 0
        y_true = (df_mci['True Label'] == 'pMCI').astype(int).values
        # Predictor: Severity (higher means more AD-like, thus pMCI)
        y_score = df_mci['Severity'].values
        
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        auc_results[var_key].append(roc_auc)
        
        # Interpolate for mean ROC
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs[var_key].append(interp_tpr)

# Calculate statistics and plot mean ROCs
for var_key, var_name in variants.items():
    mean_tpr = np.mean(tprs[var_key], axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(auc_results[var_key])
    std_auc = np.std(auc_results[var_key])
    
    plt.plot(mean_fpr, mean_tpr, color=colors[var_key],
             label=r'%s (AUC = %0.3f $\pm$ %0.3f)' % (var_name, mean_auc, std_auc),
             lw=2, alpha=.8)
    
    std_tpr = np.std(tprs[var_key], axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=colors[var_key], alpha=.2)

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='Chance', alpha=.8)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC Curve: sMCI vs pMCI Prediction (Cross-Sectional)')
plt.legend(loc="lower right")

# Calculate p-values (Paired t-test)
p_ce_03 = scipy.stats.ttest_rel(auc_results['triplet_only_margin0.3'], auc_results['ce']).pvalue
p_full_03 = scipy.stats.ttest_rel(auc_results['triplet_only_margin0.3'], auc_results['full']).pvalue
p_ce_30 = scipy.stats.ttest_rel(auc_results['triplet_only_margin3.0'], auc_results['ce']).pvalue if 'triplet_only_margin3.0' in auc_results and len(auc_results['triplet_only_margin3.0']) == 5 else 1.0
p_full_30 = scipy.stats.ttest_rel(auc_results['triplet_only_margin3.0'], auc_results['full']).pvalue if 'triplet_only_margin3.0' in auc_results and len(auc_results['triplet_only_margin3.0']) == 5 else 1.0
p_ce_ema = scipy.stats.ttest_rel(auc_results['triplet_only_ema0.5_margin0.0'], auc_results['ce']).pvalue if 'triplet_only_ema0.5_margin0.0' in auc_results and len(auc_results['triplet_only_ema0.5_margin0.0']) == 5 else 1.0
p_full_ema = scipy.stats.ttest_rel(auc_results['triplet_only_ema0.5_margin0.0'], auc_results['full']).pvalue if 'triplet_only_ema0.5_margin0.0' in auc_results and len(auc_results['triplet_only_ema0.5_margin0.0']) == 5 else 1.0

# Add p-value text to plot
textstr = f"Paired t-test (5 Folds):\nProposed(0.3) vs CE: p={p_ce_03:.4f}, vs HOPE: p={p_full_03:.4f}\nProposed(3.0) vs CE: p={p_ce_30:.4f}, vs HOPE: p={p_full_30:.4f}\nProposed(EMA) vs CE: p={p_ce_ema:.4f}, vs HOPE: p={p_full_ema:.4f}"
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "roc_smci_pmci.png"), dpi=300)
print(f"Saved ROC curve to {out_dir}")

print("=== AUC Scores ===")
for var_key, var_name in variants.items():
    print(f"{var_name}: {auc_results[var_key]} -> Mean: {np.mean(auc_results[var_key]):.4f}")

print("\n=== P-Values ===")
print(f"Proposed (0.3) vs CE Baseline: p = {p_ce_03:.5f}")
print(f"Proposed (0.3) vs HOPE (Full): p = {p_full_03:.5f}")

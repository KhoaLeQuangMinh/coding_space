"""
HOPE Model - Comprehensive Results Report
==========================================
Matches the paper's Table III, IV, and V exactly, using your 5-fold results.

Paper context:
  - ACC / AUC / F1-score / Precision / Recall all refer to the 2-class sMCI vs pMCI task.
  - Table III (Loss Ablation): tested on the TEST set, EMA σ locked to 0.9.
      Rows are cumulative: L_CE → +L_Ins2Ins → +L_Ins2Cls → +L_Cls2Cls (full HOPE)
      Uses best_2c_net.pth (model optimised for 2-class MCI task).
  - Table IV (EMA Ablation): tested on the VALIDATION set (we use test for comparison).
      Rows: No EMA / σ=0.5 / σ=0.8 / σ=0.9 / σ=0.99 / σ=0.999
      Loss locked to full. Uses best_2c_net.pth.
  - Table V Extension: shows 3-class (V1) and 4-class (V2) results of the full HOPE model.
      We report from best_3c_net and best_4c_net respectively.

Additionally, this script produces EXTENDED tables (your own contribution beyond the paper)
showing 3-class and 4-class performance across all variants.

Run:
    python analyze_results.py
    python analyze_results.py --results_dir /path/to/checkpoints
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

# ──────────────────────────────────────────────────────────────────────
# Configuration — matches your ablation scripts exactly
# ──────────────────────────────────────────────────────────────────────

LOSS_VARIANTS = ['ce', 'ins2ins', 'ins2cls', 'full', 'exclude_ins2ins', 'exclude_ins2cls']
EMA_VARIANTS  = ['None', '0.5', '0.8', '0.9', '0.99', '0.999']
N_FOLDS       = 5

# Paper-matching row labels for Table III (cumulative notation)
LOSS_LABELS = {
    'ce':      'L_CE',
    'ins2ins': '  + L_Ins2Ins',
    'ins2cls': '    + L_Ins2Cls',
    'full':    '      + L_Cls2Cls  (HOPE)',
    'exclude_ins2ins': 'Exclude Ins2Ins Ablation',
    'exclude_ins2cls': 'Exclude Ins2Cls Ablation',
}

# Paper-matching row labels for Table IV
EMA_LABELS = {
    'None':  'No EMA  (σ = 0)',
    '0.5':   'EMA     σ = 0.5',
    '0.8':   'EMA     σ = 0.8',
    '0.9':   'EMA     σ = 0.9  ★',   # paper best
    '0.99':  'EMA     σ = 0.99',
    '0.999': 'EMA     σ = 0.999',
}

# Column mapping: CSV col → display name
# 2-class metrics (paper primary metrics)
MCI_COLS = {
    'MCI Acc':  'ACC',
    'MCI AUC':  'AUC',
    'MCI F1':   'F1-score',
    'MCI Prec': 'Precision',
    'MCI SEN':  'Recall',
}
# 3-class metrics
CLS3_COLS = {
    'Acc 3-class': 'ACC',
    'MCI AUC':     'AUC',      # best proxy available
    'F1 3-class':  'F1-score',
    'MCI Prec':    'Precision',
    'MCI SEN':     'Recall',
}
# 4-class metrics
CLS4_COLS = {
    'Acc 4-class': 'ACC',
    'MCI AUC':     'AUC',
    'F1 4-class':  'F1-score',
    'MCI Prec':    'Precision',
    'MCI SEN':     'Recall',
}


# ──────────────────────────────────────────────────────────────────────
# Loader
# ──────────────────────────────────────────────────────────────────────

def load_row(base_dir, folder_prefix, test_target, fold):
    folder = f"{folder_prefix}_fold{fold}"
    path   = os.path.join(base_dir, folder, f"test_metrics_best_{test_target}.csv")
    if not os.path.exists(path):
        print(f"  [WARN] Not found: {path}", file=sys.stderr)
        return None
    df = pd.read_csv(path, index_col=0)
    label = f"Fold {fold}"
    if label not in df.index:
        label = df.index[0]
    return df.loc[label].to_dict()


def aggregate(base_dir, folder_prefix, test_target, n_folds=5):
    """Return {'mean': Series, 'std': Series, 'n': int, 'per_fold': DataFrame}"""
    rows = []
    for fold in range(1, n_folds + 1):
        r = load_row(base_dir, folder_prefix, test_target, fold)
        if r:
            rows.append(r)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    return {
        'mean':     df.mean(),
        'std':      df.std(ddof=1),
        'n':        len(rows),
        'per_fold': df.reset_index(drop=True),
    }


# ──────────────────────────────────────────────────────────────────────
# Formatting
# ──────────────────────────────────────────────────────────────────────

def pct(v):
    return v * 100

def ms(mean, std, dec=1):
    return f"{pct(mean):.{dec}f} ± {pct(std):.{dec}f}"

def get_ms(agg, col, dec=1):
    if agg is None or col not in agg['mean']:
        return "  N/A  "
    return ms(agg['mean'][col], agg['std'][col], dec)

def get_mean(agg, col, dec=1):
    """Return just the mean (no std) — matches how the original paper reports numbers."""
    if agg is None or col not in agg['mean']:
        return "  N/A"
    return f"{pct(agg['mean'][col]):.{dec}f}"

def div(width=112):
    return "─" * width

def header(title, width=112):
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)


# ──────────────────────────────────────────────────────────────────────
# Generic table printer
# ──────────────────────────────────────────────────────────────────────

def print_table(title, note, variants, row_labels, folder_prefix, test_target,
                col_map, all_agg, mean_only=False):
    """
    Print one results table.
    col_map: dict of {csv_col: display_name}
    mean_only: if True print just mean (like paper); else print mean±std
    """
    header(title)
    if note:
        print(f"  {note}\n")

    display_cols = list(col_map.values())
    csv_cols     = list(col_map.keys())

    # column widths
    var_w  = max(len(r) for r in row_labels.values()) + 2
    col_w  = 20 if not mean_only else 10

    # header row
    hdr = f"  {'Variant':<{var_w}}"
    for dc in display_cols:
        hdr += f"  {dc:>{col_w}}"
    print(hdr)
    print("  " + div(len(hdr) - 2))

    for v in variants:
        agg = all_agg[v].get(test_target)
        label = row_labels[v]
        row = f"  {label:<{var_w}}"
        for cc in csv_cols:
            val = get_mean(agg, cc) if mean_only else get_ms(agg, cc)
            row += f"  {val:>{col_w}}"
        print(row)
    print()


# ──────────────────────────────────────────────────────────────────────
# Per-fold detailed table
# ──────────────────────────────────────────────────────────────────────

def print_per_fold_table(title, variants, row_labels, folder_prefix, test_target,
                         primary_col, base_dir):
    header(f"{title}  — Per-Fold Breakdown  (metric: {primary_col})")
    fold_hdr = f"  {'Variant':<35}" + "".join(f"  {'Fold '+str(i):>10}" for i in range(1, 6)) + f"  {'Mean':>10}  {'Std':>8}"
    print(fold_hdr)
    print("  " + div(len(fold_hdr) - 2))

    for v in variants:
        label = row_labels[v]
        vals  = []
        for fold in range(1, 6):
            r = load_row(base_dir, f"ablation_{folder_prefix}_{v}", test_target, fold)
            vals.append(pct(r[primary_col]) if r and primary_col in r else float('nan'))

        arr = np.array(vals)
        row = f"  {label:<35}"
        for val in vals:
            row += f"  {val:>10.1f}" if not np.isnan(val) else f"  {'N/A':>10}"
        row += f"  {np.nanmean(arr):>10.1f}  {np.nanstd(arr, ddof=1):>8.1f}"
        print(row)
    print()


# ──────────────────────────────────────────────────────────────────────
# CSV export
# ──────────────────────────────────────────────────────────────────────

def export_summary_csv(path, variants, row_labels, test_target, col_map, all_agg):
    rows = []
    for v in variants:
        agg = all_agg[v].get(test_target)
        row = {'Variant': row_labels[v], 'Model': f"best_{test_target}_net", 'N_Folds': agg['n'] if agg else 0}
        for cc, dc in col_map.items():
            if agg and cc in agg['mean']:
                mean_val = round(pct(agg['mean'][cc]), 1)
                std_val  = round(pct(agg['std'][cc]),  1)
                row[dc] = f"{mean_val} ± {std_val}"
            else:
                row[dc] = "N/A"
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  Saved → {path}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str,
        default='/Users/khoale/Downloads/all_result/working/coding_space/Hope Implementation/checkpoints')
    parser.add_argument('--export_dir', type=str, default=None)
    args = parser.parse_args()

    base_dir   = args.results_dir
    export_dir = args.export_dir or base_dir

    print("\n" + "═" * 112)
    print("  HOPE MODEL — COMPLETE RESULTS REPORT")
    print("  All metrics are mean ± std (%) computed over 5-fold cross-validation.")
    print("  Paper metrics (ACC, AUC, F1-score, Precision, Recall) = 2-class sMCI vs pMCI task.")
    print(f"  Results dir: {base_dir}")
    print("═" * 112)

    # ── Load all aggregations ─────────────────────────────────────────
    print("\nLoading data ...")
    loss_agg = {v: {} for v in LOSS_VARIANTS}
    ema_agg  = {v: {} for v in EMA_VARIANTS}

    for v in LOSS_VARIANTS:
        for t in ['2c', '3c', '4c']:
            loss_agg[v][t] = aggregate(base_dir, f"ablation_loss_{v}", t)

    for v in EMA_VARIANTS:
        for t in ['2c', '3c', '4c']:
            ema_agg[v][t] = aggregate(base_dir, f"ablation_ema_{v}", t)

    # ═══════════════════════════════════════════════════════════════════
    # TABLE III  —  LOSS ABLATION  (paper format, 2-class primary metric)
    # best_2c_net: model saved at peak 2-class MCI validation accuracy
    # ═══════════════════════════════════════════════════════════════════
    print_table(
        title       = "TABLE III  —  LOSS COMPONENT ABLATION  (Paper Format)",
        note        = "Using best_2c_net checkpoint (model peak on sMCI vs pMCI validation).\n"
                      "  Paper reports single values; we report mean ± std over 5 folds.\n"
                      "  Cumulative: each row adds one loss component to the previous.",
        variants    = LOSS_VARIANTS,
        row_labels  = LOSS_LABELS,
        folder_prefix='loss',
        test_target = '2c',
        col_map     = MCI_COLS,
        all_agg     = loss_agg,
    )

    print_per_fold_table(
        title         = "TABLE III  —  Per-Fold ACC",
        variants      = LOSS_VARIANTS,
        row_labels    = LOSS_LABELS,
        folder_prefix = 'loss',
        test_target   = '2c',
        primary_col   = 'MCI Acc',
        base_dir      = base_dir,
    )

    # ═══════════════════════════════════════════════════════════════════
    # TABLE IV  —  EMA ABLATION  (paper format, 2-class primary metric)
    # ═══════════════════════════════════════════════════════════════════
    print_table(
        title       = "TABLE IV  —  EMA MOMENTUM ABLATION  (Paper Format)",
        note        = "Using best_2c_net checkpoint. Full HOPE loss locked.\n"
                      "  Paper reports results on internal ADNI validation set;\n"
                      "  we report on the held-out TEST set for each fold.",
        variants    = EMA_VARIANTS,
        row_labels  = EMA_LABELS,
        folder_prefix='ema',
        test_target = '2c',
        col_map     = MCI_COLS,
        all_agg     = ema_agg,
    )

    print_per_fold_table(
        title         = "TABLE IV  —  Per-Fold ACC",
        variants      = EMA_VARIANTS,
        row_labels    = EMA_LABELS,
        folder_prefix = 'ema',
        test_target   = '2c',
        primary_col   = 'MCI Acc',
        base_dir      = base_dir,
    )

    # ═══════════════════════════════════════════════════════════════════
    # TABLE V  —  EXTENSION RESULTS (3-class V1, 4-class V2)
    # ═══════════════════════════════════════════════════════════════════
    header("TABLE V  —  EXTENSION RESULTS  (Paper Format)")
    print("  V1 = HOPE trained with 3-class labels (NC / MCI / AD)  → best_3c_net checkpoint")
    print("  V2 = HOPE trained with 4-class labels (NC / sMCI / pMCI / AD)  → best_4c_net checkpoint\n")

    col_w = 20
    v_hdr = f"  {'Variant':<20}  {'Training Labels':<22}" + "".join(
        f"  {dc:>{col_w}}" for dc in ['ACC', 'AUC', 'F1-score', 'Precision', 'Recall'])
    print(v_hdr)
    print("  " + div(len(v_hdr) - 2))

    a3 = loss_agg['full'].get('3c')
    a4 = loss_agg['full'].get('4c')

    def ext_row(label, training_labels, agg, col_map):
        row = f"  {label:<20}  {training_labels:<22}"
        for cc in col_map:
            row += f"  {get_ms(agg, cc):>{col_w}}"
        return row

    # V1: 3-class — exclusively uses 3-class metrics
    v1_map = {'Acc 3-class': 'ACC', 'AUC 3-class': 'AUC', 'F1 3-class': 'F1-score', 'Prec 3-class': 'Precision', 'Recall 3-class': 'Recall'}
    # V2: 4-class — exclusively uses 4-class metrics
    v2_map = {'Acc 4-class': 'ACC', 'AUC 4-class': 'AUC', 'F1 4-class': 'F1-score', 'Prec 4-class': 'Precision', 'Recall 4-class': 'Recall'}

    print(ext_row("V1", "NC / MCI / AD",        a3, v1_map))
    print(ext_row("V2", "NC / sMCI / pMCI / AD", a4, v2_map))
    print()

    # ═══════════════════════════════════════════════════════════════════
    # TABLE V EXTENDED  —  CROSS-CHECKPOINT EVALUATION MATRIX
    # Best HOPE model (full loss, σ=0.9): each checkpoint tested on all 3 tasks
    # Rows = which task the checkpoint was SAVED for (validated on)
    # Cols = which task it is being TESTED on
    # ═══════════════════════════════════════════════════════════════════
    header("TABLE V EXTENDED  —  CROSS-CHECKPOINT EVALUATION  (Full HOPE, σ=0.9)")
    print("  Each checkpoint was saved based on its VALIDATION score for ONE task (rows).")
    print("  It is then TESTED on ALL THREE tasks (columns).")
    print("  ◀ diagonal = the task the model was optimised for.")
    print("  Off-diagonal = transfer performance to other tasks.\n")

    checkpoint_defs = [
        ('best_2c_net', '2c', 'Validated for 2-class (sMCI vs pMCI)'),
        ('best_3c_net', '3c', 'Validated for 3-class (NC/MCI/AD)'),
        ('best_4c_net', '4c', 'Validated for 4-class (NC/sMCI/pMCI/AD)'),
    ]
    cross_metrics = {
        '2c': [('MCI Acc', 'ACC'), ('MCI F1', 'F1'), ('MCI AUC', 'AUC')],
        '3c': [('Acc 3-class', 'ACC'), ('F1 3-class', 'F1')],
        '4c': [('Acc 4-class', 'ACC'), ('F1 4-class', 'F1')],
    }
    task_names = {
        '2c': '2-Class (sMCI vs pMCI)',
        '3c': '3-Class (NC/MCI/AD)',
        '4c': '4-Class (NC/sMCI/pMCI/AD)',
    }
    cell_w = 17

    # Print column header
    col_counts = {t: len(cross_metrics[t]) for t in ['2c', '3c', '4c']}
    hdr1 = f"  {'Checkpoint (Saved For)':<40}"
    for t in ['2c', '3c', '4c']:
        span = cell_w * col_counts[t] + 2 * (col_counts[t] - 1)
        hdr1 += f"  {task_names[t]:^{span}}"
    print(hdr1)

    hdr2 = f"  {'':40}"
    for t in ['2c', '3c', '4c']:
        for _, mname in cross_metrics[t]:
            hdr2 += f"  {mname:>{cell_w}}"
    print(hdr2)
    print("  " + "─" * (len(hdr2) - 2))

    cross_rows = []
    for ckpt_name, ckpt_target, row_label in checkpoint_defs:
        agg = loss_agg['full'].get(ckpt_target)
        row_str = f"  {row_label:<40}"
        csv_row = {'Checkpoint': ckpt_name, 'Optimised_For': task_names[ckpt_target]}
        for t in ['2c', '3c', '4c']:
            for col_key, mname in cross_metrics[t]:
                val_str = get_ms(agg, col_key)
                marker = ' ◀' if t == ckpt_target else '  '
                cell = val_str + marker
                row_str += f"  {cell:>{cell_w + 2}}"
                csv_row[f"{task_names[t]}_{mname}"] = val_str + (' [diag]' if t == ckpt_target else '')
        print(row_str)
        cross_rows.append(csv_row)

    print()
    print("  Interpretation: does a model optimised for task X still perform well on task Y?")
    print()

    cross_csv_path = os.path.join(export_dir, 'table5_extended_cross_checkpoint.csv')
    pd.DataFrame(cross_rows).to_csv(cross_csv_path, index=False)

    # ═══════════════════════════════════════════════════════════════════
    # EXTENDED TABLE A  —  LOSS ABLATION  across all 3 class settings

    # ═══════════════════════════════════════════════════════════════════
    header("EXTENDED TABLE A  —  LOSS ABLATION: All Classification Settings")
    print("  Shows how each loss variant performs when its best checkpoint is evaluated")
    print("  on 2-class, 3-class, and 4-class tasks. Format: mean ± std %\n")

    sub_hdrs = ['2-Class (sMCI vs pMCI)', '3-Class (NC/MCI/AD)', '4-Class (NC/sMCI/pMCI/AD)']
    sub_accs = [('MCI Acc', '2c'), ('Acc 3-class', '3c'), ('Acc 4-class', '4c')]
    sub_f1s  = [('MCI F1',  '2c'), ('F1 3-class',  '3c'), ('F1 4-class',  '4c')]
    sub_aucs = [('MCI AUC', '2c'), ('MCI AUC',     '3c'), ('MCI AUC',     '4c')]

    col_w2 = 24
    row_hdr = f"  {'Variant':<30}"
    for sh in sub_hdrs:
        row_hdr += f"  {sh:^{col_w2*3+4}}"
    print(row_hdr)

    sub_col_hdr = f"  {'':30}"
    for _ in sub_hdrs:
        sub_col_hdr += f"  {'ACC':>{col_w2}}  {'F1':>{col_w2}}  {'AUC':>{col_w2}}"
    print(sub_col_hdr)
    print("  " + div(len(sub_col_hdr) - 2))

    for v in LOSS_VARIANTS:
        row = f"  {LOSS_LABELS[v]:<30}"
        for (acc_col, t), (f1_col, _), (auc_col, __) in zip(sub_accs, sub_f1s, sub_aucs):
            agg = loss_agg[v].get(t)
            row += f"  {get_ms(agg, acc_col):>{col_w2}}  {get_ms(agg, f1_col):>{col_w2}}  {get_ms(agg, auc_col):>{col_w2}}"
        print(row)
    print()

    # ═══════════════════════════════════════════════════════════════════
    # EXTENDED TABLE B  —  EMA ABLATION  across all 3 class settings
    # ═══════════════════════════════════════════════════════════════════
    header("EXTENDED TABLE B  —  EMA ABLATION: All Classification Settings")
    print("  Same structure as Extended Table A but for EMA momentum variants.\n")

    for v in EMA_VARIANTS:
        row = f"  {EMA_LABELS[v]:<30}"
        for (acc_col, t), (f1_col, _), (auc_col, __) in zip(sub_accs, sub_f1s, sub_aucs):
            agg = ema_agg[v].get(t)
            row += f"  {get_ms(agg, acc_col):>{col_w2}}  {get_ms(agg, f1_col):>{col_w2}}  {get_ms(agg, auc_col):>{col_w2}}"
        print(row)
    print()

    # ═══════════════════════════════════════════════════════════════════
    # EXTENDED TABLE C  —  Full metric breakdown for best model (HOPE full, σ=0.9)
    # ═══════════════════════════════════════════════════════════════════
    header("EXTENDED TABLE C  —  BEST MODEL (HOPE Full, σ=0.9) — Complete Metric Breakdown")
    print("  Reports all 5 paper metrics for each of the 3 classification settings.\n")

    best_loss = loss_agg['full']  # HOPE full with σ=0.9
    all_target_defs = [
        ('2-Class (sMCI vs pMCI)', '2c', MCI_COLS),
        ('3-Class (NC / MCI / AD)', '3c', {
            'Acc 3-class': 'ACC', 'AUC 3-class': 'AUC', 'F1 3-class': 'F1-score',
            'Prec 3-class': 'Precision', 'Recall 3-class': 'Recall'}),
        ('4-Class (NC/sMCI/pMCI/AD)', '4c', {
            'Acc 4-class': 'ACC', 'AUC 4-class': 'AUC', 'F1 4-class': 'F1-score',
            'Prec 4-class': 'Precision', 'Recall 4-class': 'Recall'}),
    ]

    for label, t, cmap in all_target_defs:
        agg = best_loss.get(t)
        print(f"  ▶ {label}")
        mhdr = f"    {'Metric':<20}" + "".join(f"  {'Value (mean ± std %)':>24}")
        print(mhdr)
        print("    " + div(50))
        for cc, dc in cmap.items():
            print(f"    {dc:<20}  {get_ms(agg, cc):>24}")
        print()

    # ═══════════════════════════════════════════════════════════════════
    # CSV EXPORTS
    # ═══════════════════════════════════════════════════════════════════
    header("CSV EXPORT")

    # Table III
    export_summary_csv(
        os.path.join(export_dir, 'table3_loss_ablation_2class.csv'),
        LOSS_VARIANTS, LOSS_LABELS, '2c', MCI_COLS, loss_agg)

    # Table IV
    export_summary_csv(
        os.path.join(export_dir, 'table4_ema_ablation_2class.csv'),
        EMA_VARIANTS, EMA_LABELS, '2c', MCI_COLS, ema_agg)

    # Table V — V1 (3-class) and V2 (4-class) for full HOPE
    v1_col_map = {'Acc 3-class': 'ACC', 'AUC 3-class': 'AUC', 'F1 3-class': 'F1-score',
                  'Prec 3-class': 'Precision', 'Recall 3-class': 'Recall'}
    v2_col_map = {'Acc 4-class': 'ACC', 'AUC 4-class': 'AUC', 'F1 4-class': 'F1-score',
                  'Prec 4-class': 'Precision', 'Recall 4-class': 'Recall'}
    tableV_rows = []
    for variant_label, t, cmap in [('V1 (NC/MCI/AD)', '3c', v1_col_map),
                                    ('V2 (NC/sMCI/pMCI/AD)', '4c', v2_col_map)]:
        agg = loss_agg['full'].get(t)
        row = {'Variant': variant_label, 'Training Labels': variant_label.split('(')[1].rstrip(')'),
               'Model': f'best_{t}_net', 'N_Folds': agg['n'] if agg else 0}
        for cc, dc in cmap.items():
            if agg and cc in agg['mean']:
                row[dc] = f"{round(pct(agg['mean'][cc]),1)} ± {round(pct(agg['std'][cc]),1)}"
            else:
                row[dc] = 'N/A'
        tableV_rows.append(row)
    tableV_path = os.path.join(export_dir, 'table5_extension_3class_4class.csv')
    pd.DataFrame(tableV_rows).to_csv(tableV_path, index=False)
    print(f"  Saved → {tableV_path}")

    # Extended Table A — Loss ablation, 3-class
    export_summary_csv(
        os.path.join(export_dir, 'extended_loss_ablation_3class.csv'),
        LOSS_VARIANTS, LOSS_LABELS, '3c',
        {'Acc 3-class': 'ACC', 'AUC 3-class': 'AUC', 'F1 3-class': 'F1-score',
         'Prec 3-class': 'Precision', 'Recall 3-class': 'Recall'}, loss_agg)

    # Extended Table A — Loss ablation, 4-class
    export_summary_csv(
        os.path.join(export_dir, 'extended_loss_ablation_4class.csv'),
        LOSS_VARIANTS, LOSS_LABELS, '4c',
        {'Acc 4-class': 'ACC', 'AUC 4-class': 'AUC', 'F1 4-class': 'F1-score',
         'Prec 4-class': 'Precision', 'Recall 4-class': 'Recall'}, loss_agg)

    # Extended Table B — EMA ablation, 3-class
    export_summary_csv(
        os.path.join(export_dir, 'extended_ema_ablation_3class.csv'),
        EMA_VARIANTS, EMA_LABELS, '3c',
        {'Acc 3-class': 'ACC', 'AUC 3-class': 'AUC', 'F1 3-class': 'F1-score',
         'Prec 3-class': 'Precision', 'Recall 3-class': 'Recall'}, ema_agg)

    # Extended Table B — EMA ablation, 4-class
    export_summary_csv(
        os.path.join(export_dir, 'extended_ema_ablation_4class.csv'),
        EMA_VARIANTS, EMA_LABELS, '4c',
        {'Acc 4-class': 'ACC', 'AUC 4-class': 'AUC', 'F1 4-class': 'F1-score',
         'Prec 4-class': 'Precision', 'Recall 4-class': 'Recall'}, ema_agg)

    # Extended Table C — Best model (HOPE full) complete breakdown, all 3 tasks
    tableC_rows = []
    for task_label, t, cmap in [
        ('2-Class (sMCI vs pMCI)',   '2c', MCI_COLS),
        ('3-Class (NC/MCI/AD)',      '3c', {'Acc 3-class': 'ACC', 'AUC 3-class': 'AUC',
                                            'F1 3-class': 'F1-score', 'Prec 3-class': 'Precision',
                                            'Recall 3-class': 'Recall'}),
        ('4-Class (NC/sMCI/pMCI/AD)','4c', {'Acc 4-class': 'ACC', 'AUC 4-class': 'AUC',
                                            'F1 4-class': 'F1-score', 'Prec 4-class': 'Precision',
                                            'Recall 4-class': 'Recall'}),
    ]:
        agg = loss_agg['full'].get(t)
        row = {'Task': task_label, 'Model': f'best_{t}_net', 'N_Folds': agg['n'] if agg else 0}
        for cc, dc in cmap.items():
            if agg and cc in agg['mean']:
                row[dc] = f"{round(pct(agg['mean'][cc]),1)} ± {round(pct(agg['std'][cc]),1)}"
            else:
                row[dc] = 'N/A'
        tableC_rows.append(row)
    tableC_path = os.path.join(export_dir, 'extended_tableC_best_model_all_tasks.csv')
    pd.DataFrame(tableC_rows).to_csv(tableC_path, index=False)
    print(f"  Saved → {tableC_path}")

    print("\n" + "═" * 112)
    print("  REPORT COMPLETE.")
    print("═" * 112 + "\n")


if __name__ == '__main__':
    main()

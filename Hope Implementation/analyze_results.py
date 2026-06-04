import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

def parse_folder_name(folder_name):
    """Extracts the experiment type, variant name, and fold number from the folder name."""
    parts = folder_name.split('_')
    if "loss" in folder_name:
        experiment = "loss"
        variant = parts[2]
        fold = int(parts[-1].replace("fold", ""))
    elif "ema" in folder_name:
        experiment = "ema"
        variant = parts[2]
        fold = int(parts[-1].replace("fold", ""))
    else:
        return None, None, None
    return experiment, variant, fold

def generate_tables_and_plots(base_dir):
    best_metrics_data = []
    history_data = []
    
    print("Loading CSV files...")
    
    # 1. Traverse folders and load data
    for root, dirs, files in os.walk(base_dir):
        folder_name = os.path.basename(root)
        experiment, variant, fold = parse_folder_name(folder_name)
        
        if experiment is None:
            continue
            
        best_csv = os.path.join(root, "best_metrics.csv")
        history_csv = os.path.join(root, "history.csv")
        
        if os.path.exists(best_csv):
            df_best = pd.read_csv(best_csv)
            df_best['experiment'] = experiment
            df_best['variant'] = variant
            df_best['fold'] = fold
            best_metrics_data.append(df_best)
            
        if os.path.exists(history_csv):
            df_hist = pd.read_csv(history_csv)
            df_hist['experiment'] = experiment
            df_hist['variant'] = variant
            df_hist['fold'] = fold
            history_data.append(df_hist)
            
    if not best_metrics_data:
        print("No best_metrics.csv files found. Please check your folder path.")
        return
        
    df_all_best = pd.concat(best_metrics_data, ignore_index=True)
    df_all_hist = pd.concat(history_data, ignore_index=True)
    
    # 2. Generate Tables (Mean ± Std across 5 folds)
    metrics = ['val_acc', 'val_sen', 'val_spe', 'val_auc', 'val_f1']
    
    for exp in ['loss', 'ema']:
        print(f"\n{'='*60}\nTABLE FOR ABLATION: {exp.upper()}\n{'='*60}")
        df_exp = df_all_best[df_all_best['experiment'] == exp]
        
        if df_exp.empty:
            continue
            
        # Calculate Mean and Standard Deviation across the 5 folds
        summary = df_exp.groupby('variant')[metrics].agg(['mean', 'std'])
        
        # Format strings as "Mean ± Std" for the paper
        formatted_summary = pd.DataFrame(index=summary.index)
        for m in metrics:
            means = summary[(m, 'mean')] * 100 # Convert to percentage
            stds = summary[(m, 'std')] * 100   # Convert to percentage
            formatted_summary[m] = [f"{mean:.1f} ± {std:.1f}" for mean, std in zip(means, stds)]
            
        # Reorder variants logically for the table
        if exp == 'loss':
            order = ['ce', 'ins2ins', 'ins2cls', 'full']
        else:
            order = ['None', '0.5', '0.8', '0.9', '0.99', '0.999']
            
        formatted_summary = formatted_summary.reindex(order).dropna()
        
        # Print table to console and save as CSV
        print(formatted_summary.to_markdown())
        table_filename = f"table_{exp}_ablation.csv"
        formatted_summary.to_csv(table_filename)
        print(f"\nSaved table to -> {table_filename}")
        
    # 3. Plot Learning Curves using Seaborn
    print("\nGenerating Learning Curve Plots...")
    sns.set_theme(style="whitegrid")
    
    for exp in ['loss', 'ema']:
        df_exp_hist = df_all_hist[df_all_hist['experiment'] == exp]
        if df_exp_hist.empty:
            continue
            
        plt.figure(figsize=(10, 6))
        # Seaborn automatically averages the 5 folds at each epoch and draws a smooth line!
        sns.lineplot(data=df_exp_hist, x='epoch', y='val_acc', hue='variant', errorbar=None, linewidth=2.5)
        
        plt.title(f'Validation Accuracy Learning Curve ({exp.upper()} Ablation)', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Validation Accuracy', fontsize=14)
        plt.legend(title='Variant', title_fontsize='13', fontsize='12')
        plt.tight_layout()
        
        plot_filename = f"plot_{exp}_learning_curve.png"
        plt.savefig(plot_filename, dpi=300)
        print(f"Saved plot to -> {plot_filename}")
        plt.close()

if __name__ == "__main__":
    # Expects the user to pass the path to the folder containing all the CSV folders
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <path_to_checkpoints_folder>")
        sys.exit(1)
        
    base_dir = sys.argv[1]
    generate_tables_and_plots(base_dir)

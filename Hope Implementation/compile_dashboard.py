import os
import shutil
import json
import pandas as pd

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
base_dir = '/Users/khoale/Desktop/Coding_Space/Hope Implementation'
downloads_dir = '/Users/khoale/Downloads'
dest_dir = os.path.join(downloads_dir, 'plot_display')
plots_dest = os.path.join(dest_dir, 'plots')

# Source directories
src_tsne_combined = os.path.join(downloads_dir, 'analysis_output_tSNE')
src_tsne_folds = os.path.join(downloads_dir, 'analysis_output_tSNE/fold_plots')
src_pca_folds = os.path.join(downloads_dir, 'analysis_output_tSNE/plot_pca_folds')
src_kde_combined = os.path.join(downloads_dir, 'analysis_output/latent_plots')
src_kde_folds = os.path.join(downloads_dir, 'analysis_output_tSNE/plot_kde_folds')
src_merged = os.path.join(downloads_dir, 'ablation_result/plots')
src_checkpoints = os.path.join(downloads_dir, 'ablation_result/working/coding_space/Hope Implementation/checkpoints')

# Comparative sources
src_umap = os.path.join(downloads_dir, 'ablation_result/proposed')
src_dist_box = os.path.join(downloads_dir, 'ablation_result/statistical_proof')
src_dist_scatter = os.path.join(downloads_dir, 'ablation_result/proposed')

# Subdirectories in destination
subdirs = {
    'tsne_combined': os.path.join(plots_dest, 'tsne_combined'),
    'tsne_folds': os.path.join(plots_dest, 'tsne_folds'),
    'pca_folds': os.path.join(plots_dest, 'pca_folds'),
    'kde_combined': os.path.join(plots_dest, 'kde_combined'),
    'kde_folds': os.path.join(plots_dest, 'kde_folds'),
    'merged': os.path.join(plots_dest, 'merged'),
    'confusion_matrices': os.path.join(plots_dest, 'confusion_matrices'),
    'comparative': os.path.join(plots_dest, 'comparative')
}

# Loss variants lists & labels
LOSS_VARIANTS = [
    'ce', 'ins2ins', 'ins2cls', 'full', 'exclude_ins2ins', 'exclude_ins2cls',
    'exp_triplet_ins2cls', 'triplet_only', 'hierarchical_triplet_only',
    'exp_hierarchical_triplet_ins2cls', 'full_4class', 'exp_triplet_ins2cls_4class',
    'hierarchical_triplet_only_4class', 'qwk_hierarchical_triplet_4class',
    'exp_3pole_local', 'exp_3pole_global', '3pole_local_only', '3pole_global_only'
]

LOSS_LABELS = {
    'ce': 'L_CE',
    'ins2ins': 'L_CE + L_Ins2Ins',
    'ins2cls': 'L_CE + L_Ins2Ins + L_Ins2Cls',
    'full': 'L_CE + L_Ins2Ins + L_Ins2Cls + L_Cls2Cls  (HOPE)',
    'exclude_ins2ins': 'L_CE + L_Ins2Cls + L_Cls2Cls (Exclude Ins2Ins)',
    'exclude_ins2cls': 'L_CE + L_Ins2Ins + L_Cls2Cls (Exclude Ins2Cls)',
    'exp_triplet_ins2cls': 'L_CE + L_Ins2Ins + L_Triplet + L_Cls2Cls',
    'triplet_only': 'L_CE + L_Triplet',
    'hierarchical_triplet_only': 'L_CE + L_Hierarchical_Triplet',
    'exp_hierarchical_triplet_ins2cls': 'L_CE + L_Ins2Ins + L_Hierarchical_Triplet + L_Cls2Cls',
    'full_4class': 'L_CE + L_Ins2Ins + L_Ins2Cls + L_Cls2Cls (4-Class)',
    'exp_triplet_ins2cls_4class': 'L_CE + L_Ins2Ins + L_Triplet + L_Cls2Cls (4-Class)',
    'hierarchical_triplet_only_4class': 'L_CE + L_Hierarchical_Triplet (4-Class)',
    'qwk_hierarchical_triplet_4class': 'L_QWK + L_Hierarchical_Triplet (4-Class)',
    'exp_3pole_local': 'L_CE + L_Ins2Ins + L_3Pole_Triplet (Local) + L_Cls2Cls',
    'exp_3pole_global': 'L_CE + L_Ins2Ins + L_3Pole_Triplet (Global) + L_Cls2Cls',
    '3pole_local_only': 'L_CE + L_3Pole_Triplet (Local) Only',
    '3pole_global_only': 'L_CE + L_3Pole_Triplet (Global) Only'
}

CHECKPOINTS = ['best_2c_net', 'best_3c_net', 'best_4c_net']
FOLDS = [1, 2, 3, 4, 5]

# ──────────────────────────────────────────────────────────────────────
# 1. Setup Directories
# ──────────────────────────────────────────────────────────────────────
print("Setting up plot_display directories...")
os.makedirs(dest_dir, exist_ok=True)
os.makedirs(plots_dest, exist_ok=True)
for sd_path in subdirs.values():
    os.makedirs(sd_path, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# 2. File Mover Helper
# ──────────────────────────────────────────────────────────────────────
def move_file(src, dest):
    """Moves a file if it exists, overwriting the destination if present. 
    If source does not exist, checks if destination exists (for subsequent runs)."""
    if os.path.exists(src):
        if os.path.exists(dest):
            os.remove(dest)
        shutil.move(src, dest)
        return True
    elif os.path.exists(dest):
        return True
    return False

# Record what files are present
file_registry = {v: {ckpt: {'combined': {}, 'folds': {f: {} for f in FOLDS}} for ckpt in CHECKPOINTS} for v in LOSS_VARIANTS}
comparative_registry = {'umap': {}, 'distance_scatter': {}, 'boxplots': False}

# ──────────────────────────────────────────────────────────────────────
# 3. Move and Register Files
# ──────────────────────────────────────────────────────────────────────
print("Moving files into plot_display...")

# Move comparative files
for ckpt in CHECKPOINTS:
    for fold in FOLDS:
        # UMAP
        umap_name = f"UMAP_{ckpt}_fold{fold}.png"
        src_u = os.path.join(src_umap, umap_name)
        dest_u = os.path.join(subdirs['comparative'], umap_name)
        if move_file(src_u, dest_u):
            if ckpt not in comparative_registry['umap']:
                comparative_registry['umap'][ckpt] = {}
            comparative_registry['umap'][ckpt][fold] = f"plots/comparative/{umap_name}"

# Distance Scatter
for fold in FOLDS:
    sc_name = f"Distance_Scatter_best_2c_net_fold{fold}.png"
    src_sc = os.path.join(src_dist_scatter, sc_name)
    dest_sc = os.path.join(subdirs['comparative'], sc_name)
    if move_file(src_sc, dest_sc):
        comparative_registry['distance_scatter'][fold] = f"plots/comparative/{sc_name}"

# Boxplots
bp_name = "quantitative_distance_boxplots.png"
src_bp = os.path.join(src_dist_box, bp_name)
dest_bp = os.path.join(subdirs['comparative'], bp_name)
if move_file(src_bp, dest_bp):
    comparative_registry['boxplots'] = f"plots/comparative/{bp_name}"

# Move variant files
for v in LOSS_VARIANTS:
    for ckpt in CHECKPOINTS:
        # Combined t-SNE
        tsne_name = f"tsne_{v}_{ckpt}.png"
        src_tc = os.path.join(src_tsne_combined, tsne_name)
        dest_tc = os.path.join(subdirs['tsne_combined'], tsne_name)
        if move_file(src_tc, dest_tc):
            file_registry[v][ckpt]['combined']['tsne'] = f"plots/tsne_combined/{tsne_name}"
            
        # Combined KDE
        kde_name = f"latent_kde_{v}_{ckpt}.png"
        # Check in src_kde_combined or src_kde_folds
        src_kc = os.path.join(src_kde_combined, kde_name)
        if not os.path.exists(src_kc):
            src_kc = os.path.join(src_kde_folds, kde_name)
        dest_kc = os.path.join(subdirs['kde_combined'], kde_name)
        if move_file(src_kc, dest_kc):
            file_registry[v][ckpt]['combined']['kde'] = f"plots/kde_combined/{kde_name}"
            
        # Merged
        merged_name = f"merged_{v}_{ckpt}.png"
        src_m = os.path.join(src_merged, merged_name)
        dest_m = os.path.join(subdirs['merged'], merged_name)
        if move_file(src_m, dest_m):
            file_registry[v][ckpt]['combined']['merged'] = f"plots/merged/{merged_name}"

        # Per-fold files
        for fold in FOLDS:
            # Per-fold t-SNE
            ft_name = f"tsne_{v}_{ckpt}_fold{fold}.png"
            src_ft = os.path.join(src_tsne_folds, ft_name)
            dest_ft = os.path.join(subdirs['tsne_folds'], ft_name)
            if move_file(src_ft, dest_ft):
                file_registry[v][ckpt]['folds'][fold]['tsne'] = f"plots/tsne_folds/{ft_name}"
                
            # Per-fold PCA
            fp_name = f"pca_{v}_{ckpt}_fold{fold}.png"
            src_fp = os.path.join(src_pca_folds, fp_name)
            dest_fp = os.path.join(subdirs['pca_folds'], fp_name)
            if move_file(src_fp, dest_fp):
                file_registry[v][ckpt]['folds'][fold]['pca'] = f"plots/pca_folds/{fp_name}"
                
            # Per-fold KDE
            fk_name = f"latent_kde_{v}_{ckpt}_fold{fold}.png"
            src_fk = os.path.join(src_kde_folds, fk_name)
            dest_fk = os.path.join(subdirs['kde_folds'], fk_name)
            if move_file(src_fk, dest_fk):
                file_registry[v][ckpt]['folds'][fold]['kde'] = f"plots/kde_folds/{fk_name}"

            # Confusion Matrices (if available, e.g., 3-pole variants)
            # Find any CM matching the variant and fold
            fold_ckpt_dir = os.path.join(src_checkpoints, f"ablation_loss_{v}_fold{fold}")
            if os.path.exists(fold_ckpt_dir):
                for f_name in os.listdir(fold_ckpt_dir):
                    if "confusion_matrix" in f_name and f_name.endswith(".png"):
                        # Extract train_mode and ckpt_target from the filename
                        # format: ablation_loss_{v}_confusion_matrix_{train_mode}_best_{ckpt_target}.png
                        parts = f_name.split("_confusion_matrix_")
                        if len(parts) > 1:
                            sub_parts = parts[1].split("_best_")
                            if len(sub_parts) > 1:
                                train_mode = sub_parts[0]  # '3c' or '4c'
                                ckpt_target = sub_parts[1].replace(".png", "")  # '2c', '3c', '4c'
                                
                                # Verify if this CM corresponds to our current checkpoint
                                # best_2c_net -> 2c, best_3c_net -> 3c, best_4c_net -> 4c
                                if (ckpt == 'best_2c_net' and ckpt_target == '2c') or \
                                   (ckpt == 'best_3c_net' and ckpt_target == '3c') or \
                                   (ckpt == 'best_4c_net' and ckpt_target == '4c'):
                                    
                                    src_cm = os.path.join(fold_ckpt_dir, f_name)
                                    dest_cm_name = f"confusion_matrix_{v}_fold{fold}_{train_mode}_best_{ckpt_target}.png"
                                    dest_cm = os.path.join(subdirs['confusion_matrices'], dest_cm_name)
                                    
                                    if move_file(src_cm, dest_cm):
                                        if 'cms' not in file_registry[v][ckpt]['folds'][fold]:
                                            file_registry[v][ckpt]['folds'][fold]['cms'] = []
                                        file_registry[v][ckpt]['folds'][fold]['cms'].append({
                                            'train_mode': train_mode,
                                            'path': f"plots/confusion_matrices/{dest_cm_name}"
                                        })

# ──────────────────────────────────────────────────────────────────────
# 4. Parse CSV Performance Metrics
# ──────────────────────────────────────────────────────────────────────
print("Parsing performance metrics from CSV tables...")
metrics_registry = {v: {'2c': {}, '3c': {}, '4c': {}} for v in LOSS_VARIANTS}

csv_2c_path = os.path.join(base_dir, 'table3_loss_ablation_2class.csv')
csv_3c_path = os.path.join(base_dir, 'extended_loss_ablation_3class.csv')
csv_4c_path = os.path.join(base_dir, 'extended_loss_ablation_4class.csv')

def parse_metrics_csv(csv_path, target_key):
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Skipping metrics parser.")
        return
        
    df = pd.read_csv(csv_path)
    
    # NORM_MAP mapping the normalized variant strings to the LOSS_VARIANTS keys
    norm_map = {
        'lce': 'ce',
        'lins2ins': 'ins2ins',
        'lins2cls': 'ins2cls',
        'lcls2clshope': 'full',
        'excludeins2insablation': 'exclude_ins2ins',
        'excludeins2clsablation': 'exclude_ins2cls',
        'tripletins2clspoles': 'exp_triplet_ins2cls',
        'cetripletonly': 'triplet_only',
        'cehierarchicaltripletonly': 'hierarchical_triplet_only',
        'hierarchicaltripletins2cls': 'exp_hierarchical_triplet_ins2cls',
        'hope4classnoema': 'full_4class',
        'tripletins2cls4class': 'exp_triplet_ins2cls_4class',
        'hierarchicaltriplet4class': 'hierarchical_triplet_only_4class',
        'qwkhierarchicaltriplet4class': 'qwk_hierarchical_triplet_4class',
        '3poletripletlocal': 'exp_3pole_local',
        '3poletripletglobal': 'exp_3pole_global',
        'ce3poletripletlocalonly': '3pole_local_only',
        'ce3poletripletglobalonly': '3pole_global_only'
    }

    for _, row in df.iterrows():
        raw_var = str(row['Variant'])
        # Normalize: strip spaces, lowercase, remove +, -, _, (, ), and spaces
        norm_var = raw_var.strip().replace(" ", "").replace("+", "").replace("-", "").replace("_", "").replace("(", "").replace(")", "").replace("★", "").lower()
        
        # Try to find match in norm_map
        v = norm_map.get(norm_var)
        if v:
            def clean_val(val):
                val_str = str(val).strip()
                if val_str.lower() in ('nan', 'none', 'null', ''):
                    return 'N/A'
                return val_str
            
            metrics_registry[v][target_key] = {
                'ACC': clean_val(row.get('ACC', 'N/A')),
                'QWK': clean_val(row.get('QWK', 'N/A')),
                'AUC': clean_val(row.get('AUC', 'N/A')),
                'F1': clean_val(row.get('F1-score', 'N/A')),
                'Precision': clean_val(row.get('Precision', 'N/A')),
                'Recall': clean_val(row.get('Recall', 'N/A')),
            }
        else:
            print(f"Warning: Could not match CSV variant '{raw_var}' (normalized: '{norm_var}')")

parse_metrics_csv(csv_2c_path, '2c')
parse_metrics_csv(csv_3c_path, '3c')
parse_metrics_csv(csv_4c_path, '4c')

# ──────────────────────────────────────────────────────────────────────
# 5. Build index.html
# ──────────────────────────────────────────────────────────────────────
print("Generating index.html dashboard...")

html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HOPE Latent Space Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0f172a;
            --surface-color: #1e293b;
            --surface-hover: #334155;
            --border-color: #475569;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --primary: #6366f1;
            --primary-hover: #4f46e5;
        }
        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            margin: 0;
            padding: 0;
            display: flex;
            height: 100vh;
            overflow: hidden;
        }
        
        /* Sidebar Layout */
        .sidebar {
            width: 320px;
            background-color: var(--surface-color);
            border-right: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            height: 100%;
        }
        .sidebar-header {
            padding: 24px;
            border-bottom: 1px solid var(--border-color);
            font-weight: 700;
            font-size: 1.25rem;
            letter-spacing: -0.025em;
            background: linear-gradient(90deg, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .sidebar-menu {
            flex: 1;
            overflow-y: auto;
            padding: 16px;
        }
        .menu-section-title {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin: 16px 0 8px 8px;
            font-weight: 600;
        }
        .menu-item {
            padding: 10px 14px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.875rem;
            font-weight: 500;
            margin-bottom: 4px;
            color: var(--text-muted);
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .menu-item:hover {
            background-color: var(--surface-hover);
            color: var(--text-main);
        }
        .menu-item.active {
            background-color: var(--primary);
            color: var(--text-main);
        }
        .menu-badge {
            font-size: 0.7rem;
            padding: 2px 6px;
            background-color: rgba(255,255,255,0.15);
            border-radius: 4px;
            font-weight: 600;
            color: var(--text-main);
        }
        
        /* Main Workspace */
        .workspace {
            flex: 1;
            display: flex;
            flex-direction: column;
            height: 100%;
            overflow-y: auto;
            padding: 32px;
            box-sizing: border-box;
        }
        
        .header-section {
            margin-bottom: 24px;
        }
        .title {
            font-size: 2rem;
            font-weight: 700;
            margin: 0 0 8px 0;
            letter-spacing: -0.025em;
        }
        .subtitle {
            color: var(--text-muted);
            margin: 0;
            font-size: 1rem;
        }
        
        /* Filter Controls */
        .filters {
            display: flex;
            gap: 16px;
            background-color: var(--surface-color);
            padding: 16px 24px;
            border-radius: 12px;
            border: 1px solid var(--border-color);
            margin-bottom: 28px;
            align-items: center;
        }
        .filter-group {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }
        .filter-label {
            font-size: 0.75rem;
            color: var(--text-muted);
            font-weight: 600;
            text-transform: uppercase;
        }
        .filter-select {
            background-color: var(--bg-color);
            color: var(--text-main);
            border: 1px solid var(--border-color);
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 0.875rem;
            cursor: pointer;
            outline: none;
        }
        .filter-select:focus {
            border-color: var(--primary);
        }
        
        /* Metrics Table */
        .metrics-card {
            background-color: var(--surface-color);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 28px;
        }
        .section-title {
            font-size: 1.125rem;
            font-weight: 600;
            margin: 0 0 16px 0;
            border-left: 4px solid var(--primary);
            padding-left: 12px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
            gap: 20px;
        }
        table.metrics-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.8rem;
        }
        table.metrics-table th, table.metrics-table td {
            text-align: center;
            padding: 8px 4px;
            border-bottom: 1px solid var(--border-color);
            white-space: nowrap;
        }
        table.metrics-table th {
            color: var(--text-muted);
            font-weight: 600;
        }
        table.metrics-table td.val {
            font-weight: 600;
            color: #a5b4fc;
        }
        
        /* Grid Plot Container */
        .plots-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
            gap: 24px;
            margin-bottom: 28px;
        }
        .overview-section .plots-grid {
            grid-template-columns: 1fr;
        }
        .overview-section .plot-img {
            max-height: none;
            width: 100%;
        }
        .overview-section .plot-img-container {
            cursor: default;
        }
        .overview-section {
            margin-bottom: 48px;
        }
        .plot-card {
            background-color: var(--surface-color);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            transition: transform 0.2s;
        }
        .plot-card:hover {
            transform: translateY(-2px);
        }
        .plot-title {
            padding: 14px 18px;
            font-size: 0.875rem;
            font-weight: 600;
            border-bottom: 1px solid var(--border-color);
            background-color: rgba(0,0,0,0.1);
        }
        .plot-img-container {
            flex: 1;
            background-color: #000;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 12px;
            cursor: zoom-in;
            min-height: 240px;
        }
        .plot-img {
            max-width: 100%;
            max-height: 380px;
            object-fit: contain;
            border-radius: 6px;
        }
        
        .no-data-msg {
            color: var(--text-muted);
            padding: 40px;
            text-align: center;
            font-size: 0.9rem;
            font-style: italic;
        }
        
        /* Lightbox Overlay */
        .lightbox {
            display: none;
            position: fixed;
            z-index: 9999;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background-color: rgba(15, 23, 42, 0.95);
            align-items: center;
            justify-content: center;
            cursor: zoom-out;
        }
        .lightbox-img {
            max-width: 90%;
            max-height: 90%;
            object-fit: contain;
            border-radius: 8px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        }
        .lightbox-close {
            position: absolute;
            top: 24px;
            right: 24px;
            background: rgba(255,255,255,0.1);
            border: none;
            color: var(--text-main);
            padding: 10px 16px;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
        }
    </style>
</head>
<body>

    <!-- SIDEBAR -->
    <div class="sidebar">
        <div class="sidebar-header">HOPE Latent Space Dashboard</div>
        <div class="sidebar-menu">
            <div class="menu-section-title">Comparative Analysis</div>
            <div class="menu-item active" onclick="switchTab('overview')">
                <span>Overview (UMAP & Distances)</span>
            </div>
            
            <div class="menu-section-title">Ablation Experiments</div>
            <div id="experiments-list"></div>
        </div>
    </div>

    <!-- MAIN WORKSPACE -->
    <div class="workspace">
        
        <!-- Header -->
        <div class="header-section">
            <h1 class="title" id="ws-title">Overview</h1>
            <p class="subtitle" id="ws-subtitle">Comparing latent space distribution across different models and folds.</p>
        </div>

        <!-- Overview Comparison Controls (Overview only) -->
        <div class="filters" id="overview-comparison-panel" style="display:none; flex-direction:column; gap:20px; align-items: stretch;">
            <!-- Row 1: Select Plot Type, Checkpoint, Fold -->
            <div style="display: flex; gap: 24px; flex-wrap: wrap;">
                <div class="filter-group">
                    <span class="filter-label">Comparison Plot Type</span>
                    <select class="filter-select" id="comp-plot-type" onchange="onFilterChange()">
                        <option value="umap" selected>UMAP Projection</option>
                        <option value="tsne">t-SNE Projection</option>
                        <option value="pca">PCA Projection (Per-Fold only)</option>
                        <option value="kde">KDE Severity Density</option>
                        <option value="dist_scatter">Distance Scatter (CN vs AD)</option>
                        <option value="dist_box">AD Prototype Distance Boxplot</option>
                        <option value="confusion_matrix">Confusion Matrix</option>
                        <option value="merged">Aggregate Merged t-SNE & KDE</option>
                    </select>
                </div>
                <div class="filter-group" id="comp-ckpt-group">
                    <span class="filter-label">Model Checkpoint</span>
                    <select class="filter-select" id="comp-ckpt" onchange="onFilterChange()">
                        <option value="best_2c_net">best_2c_net (sMCI vs pMCI)</option>
                        <option value="best_3c_net" selected>best_3c_net (3-Class)</option>
                        <option value="best_4c_net">best_4c_net (4-Class)</option>
                    </select>
                </div>
                <div class="filter-group" id="comp-fold-group">
                    <span class="filter-label">Evaluation Fold</span>
                    <select class="filter-select" id="comp-fold" onchange="onFilterChange()">
                        <option value="1" selected>Fold 1</option>
                        <option value="2">Fold 2</option>
                        <option value="3">Fold 3</option>
                        <option value="4">Fold 4</option>
                        <option value="5">Fold 5</option>
                    </select>
                </div>
            </div>
            
            <!-- Row 2: Select Experiments (up to 5) -->
            <div class="filter-group">
                <span class="filter-label" style="display: flex; align-items: center; gap: 8px;">
                    Select Experiments to Compare (Select up to 5) 
                    <span id="checked-count" style="color:#818cf8; font-weight:700;">(3/5)</span>
                </span>
                <div id="comp-experiments-checkboxes" style="display:grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 10px; margin-top: 10px; background-color: var(--bg-color); padding: 16px; border-radius: 8px; border: 1px solid var(--border-color);">
                </div>
            </div>
        </div>

        <!-- Global Filters (Checkpoint / Fold - Individual Experiment Tabs only) -->
        <div class="filters" id="filters-container" style="display:none;">
            <div class="filter-group">
                <span class="filter-label">Model Target</span>
                <select class="filter-select" id="select-ckpt" onchange="onFilterChange()">
                    <option value="best_2c_net">sMCI vs pMCI Checkpoint (best_2c_net)</option>
                    <option value="best_3c_net" selected>3-Class Target Checkpoint (best_3c_net)</option>
                    <option value="best_4c_net">4-Class Target Checkpoint (best_4c_net)</option>
                </select>
            </div>
            
            <div class="filter-group">
                <span class="filter-label">Evaluation Fold</span>
                <select class="filter-select" id="select-fold" onchange="onFilterChange()">
                    <option value="1" selected>Fold 1</option>
                    <option value="2">Fold 2</option>
                    <option value="3">Fold 3</option>
                    <option value="4">Fold 4</option>
                    <option value="5">Fold 5</option>
                </select>
            </div>
        </div>

        <!-- Metrics Table (Shown only for experiments) -->
        <div class="metrics-card" id="metrics-card" style="display:none;">
            <h3 class="section-title">Ablation Performance Results</h3>
            <div class="metrics-grid">
                <div>
                    <h4 style="margin:0 0 10px 0; color: var(--text-muted); font-size:0.875rem;">2-Class (sMCI vs pMCI) Evaluation</h4>
                    <table class="metrics-table">
                        <thead>
                            <tr><th>ACC</th><th>QWK</th><th>AUC</th><th>F1-score</th><th>Precision</th><th>Recall</th></tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td class="val" id="m-2c-acc">-</td>
                                <td class="val" id="m-2c-qwk">-</td>
                                <td class="val" id="m-2c-auc">-</td>
                                <td class="val" id="m-2c-f1">-</td>
                                <td class="val" id="m-2c-prec">-</td>
                                <td class="val" id="m-2c-rec">-</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                <div>
                    <h4 style="margin:0 0 10px 0; color: var(--text-muted); font-size:0.875rem;">3-Class (NC / MCI / AD) Evaluation</h4>
                    <table class="metrics-table">
                        <thead>
                            <tr><th>ACC</th><th>QWK</th><th>AUC</th><th>F1-score</th><th>Precision</th><th>Recall</th></tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td class="val" id="m-3c-acc">-</td>
                                <td class="val" id="m-3c-qwk">-</td>
                                <td class="val" id="m-3c-auc">-</td>
                                <td class="val" id="m-3c-f1">-</td>
                                <td class="val" id="m-3c-prec">-</td>
                                <td class="val" id="m-3c-rec">-</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                <div>
                    <h4 style="margin:0 0 10px 0; color: var(--text-muted); font-size:0.875rem;">4-Class (NC / sMCI / pMCI / AD) Evaluation</h4>
                    <table class="metrics-table">
                        <thead>
                            <tr><th>ACC</th><th>QWK</th><th>AUC</th><th>F1-score</th><th>Precision</th><th>Recall</th></tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td class="val" id="m-4c-acc">-</td>
                                <td class="val" id="m-4c-qwk">-</td>
                                <td class="val" id="m-4c-auc">-</td>
                                <td class="val" id="m-4c-f1">-</td>
                                <td class="val" id="m-4c-prec">-</td>
                                <td class="val" id="m-4c-rec">-</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- Plots Container -->
        <h3 class="section-title" id="plots-section-title">Overview Projections</h3>
        <div class="plots-grid" id="plots-container"></div>
        
    </div>

    <!-- LIGHTBOX OVERLAY -->
    <div class="lightbox" id="lightbox" onclick="closeLightbox()">
        <button class="lightbox-close">Close</button>
        <img class="lightbox-img" id="lightbox-img" src="" alt="Zoomed Plot">
    </div>

    <!-- DATA REGISTRIES -->
    <script>
        const fileRegistry = __FILE_REGISTRY__;
        const comparativeRegistry = __COMPARATIVE_REGISTRY__;
        const metricsRegistry = __METRICS_REGISTRY__;
        const lossLabels = __LOSS_LABELS__;
        const lossVariants = __LOSS_VARIANTS__;
        
        let currentTab = 'overview'; // 'overview' or variant key
        
        // ────────────────────────────────────────────────────────
        // Initialize Sidebar
        // ────────────────────────────────────────────────────────
        const expList = document.getElementById('experiments-list');
        lossVariants.forEach(v => {
            const div = document.createElement('div');
            div.className = 'menu-item';
            div.id = 'menu-' + v;
            div.onclick = () => switchTab(v);
            
            // Check what outputs are available
            let badge = '';
            // Basic badge logic
            if (v.includes('3pole')) {
                badge = '<span class="menu-badge">3-Pole</span>';
            } else if (v === 'full' || v === 'ce') {
                badge = '<span class="menu-badge">Core</span>';
            }
            
            div.innerHTML = `<span>${lossLabels[v]}</span>${badge}`;
            expList.appendChild(div);
        });

        // Initialize Comparison Checkboxes
        function initComparisonCheckboxes() {
            const container = document.getElementById('comp-experiments-checkboxes');
            container.innerHTML = '';
            const defaultSelected = ['ce', 'ins2cls', 'full'];
            
            lossVariants.forEach(v => {
                const label = document.createElement('label');
                label.style.display = 'flex';
                label.style.alignItems = 'center';
                label.style.gap = '8px';
                label.style.fontSize = '0.85rem';
                label.style.cursor = 'pointer';
                label.style.color = 'var(--text-main)';
                
                const cb = document.createElement('input');
                cb.type = 'checkbox';
                cb.value = v;
                cb.name = 'comp-exp';
                cb.style.cursor = 'pointer';
                if (defaultSelected.includes(v)) {
                    cb.checked = true;
                }
                
                cb.onchange = () => {
                    const checked = document.querySelectorAll('input[name="comp-exp"]:checked');
                    if (checked.length > 5) {
                        cb.checked = false;
                        alert('You can select up to 5 experiments to compare.');
                        return;
                    }
                    updateCheckedCount();
                    renderWorkspace();
                };
                
                label.appendChild(cb);
                const span = document.createElement('span');
                span.innerText = lossLabels[v];
                label.appendChild(span);
                
                container.appendChild(label);
            });
            updateCheckedCount();
        }

        function updateCheckedCount() {
            const checked = document.querySelectorAll('input[name="comp-exp"]:checked');
            document.getElementById('checked-count').innerText = `(${checked.length}/5)`;
        }

        function updateSelectorVisibility() {
            const plotType = document.getElementById('comp-plot-type').value;
            const ckptGroup = document.getElementById('comp-ckpt-group');
            const foldGroup = document.getElementById('comp-fold-group');
            
            if (plotType === 'dist_box') {
                ckptGroup.style.display = 'none';
                foldGroup.style.display = 'none';
            } else if (plotType === 'merged') {
                ckptGroup.style.display = 'block';
                foldGroup.style.display = 'none';
            } else {
                ckptGroup.style.display = 'block';
                foldGroup.style.display = 'block';
            }
        }

        // ────────────────────────────────────────────────────────
        // Tab Switch Logic
        // ────────────────────────────────────────────────────────
        function switchTab(tabId) {
            currentTab = tabId;
            
            // Update active menu style
            document.querySelectorAll('.menu-item').forEach(el => el.classList.remove('active'));
            if (tabId === 'overview') {
                document.querySelector('.sidebar-menu .menu-item').classList.add('active');
                document.getElementById('ws-title').innerText = "Comparison Workspace";
                document.getElementById('ws-subtitle').innerText = "Select up to 5 experiments and a plot type below to compare their latent space properties side-by-side.";
                document.getElementById('metrics-card').style.display = 'none';
                document.getElementById('plots-section-title').innerText = "Side-by-Side Comparison Grid";
                
                // Show overview controls, hide individual controls
                document.getElementById('overview-comparison-panel').style.display = 'flex';
                document.getElementById('filters-container').style.display = 'none';
                
                updateSelectorVisibility();
            } else {
                document.getElementById('menu-' + tabId).classList.add('active');
                document.getElementById('ws-title').innerText = lossLabels[tabId];
                document.getElementById('ws-subtitle').innerText = "Latent Severity density distributions, t-SNE / PCA / UMAP projections, distances, and confusion matrices.";
                document.getElementById('plots-section-title').innerText = "Experiment Performance Plots";
                
                // Update metrics table
                const m2 = metricsRegistry[tabId]['2c'] || {};
                const m3 = metricsRegistry[tabId]['3c'] || {};
                const m4 = metricsRegistry[tabId]['4c'] || {};
                
                document.getElementById('m-2c-acc').innerText = m2.ACC || 'N/A';
                document.getElementById('m-2c-qwk').innerText = m2.QWK || 'N/A';
                document.getElementById('m-2c-auc').innerText = m2.AUC || 'N/A';
                document.getElementById('m-2c-f1').innerText = m2.F1 || 'N/A';
                document.getElementById('m-2c-prec').innerText = m2.Precision || 'N/A';
                document.getElementById('m-2c-rec').innerText = m2.Recall || 'N/A';
                
                document.getElementById('m-3c-acc').innerText = m3.ACC || 'N/A';
                document.getElementById('m-3c-qwk').innerText = m3.QWK || 'N/A';
                document.getElementById('m-3c-auc').innerText = m3.AUC || 'N/A';
                document.getElementById('m-3c-f1').innerText = m3.F1 || 'N/A';
                document.getElementById('m-3c-prec').innerText = m3.Precision || 'N/A';
                document.getElementById('m-3c-rec').innerText = m3.Recall || 'N/A';
                
                document.getElementById('m-4c-acc').innerText = m4.ACC || 'N/A';
                document.getElementById('m-4c-qwk').innerText = m4.QWK || 'N/A';
                document.getElementById('m-4c-auc').innerText = m4.AUC || 'N/A';
                document.getElementById('m-4c-f1').innerText = m4.F1 || 'N/A';
                document.getElementById('m-4c-prec').innerText = m4.Precision || 'N/A';
                document.getElementById('m-4c-rec').innerText = m4.Recall || 'N/A';
                
                document.getElementById('metrics-card').style.display = 'block';
                
                // Hide overview controls, show individual controls
                document.getElementById('overview-comparison-panel').style.display = 'none';
                document.getElementById('filters-container').style.display = 'flex';
            }
            
            renderWorkspace();
        }

        // Resolve plot path helper
        function resolvePlotPath(v, plotType, ckpt, fold) {
            switch(plotType) {
                case 'umap':
                    return `plots/umap_individual/umap_${v}_${ckpt}_fold${fold}.png`;
                case 'dist_scatter':
                    return `plots/distance_scatter_individual/distance_scatter_${v}_fold${fold}.png`;
                case 'dist_box':
                    return `plots/distance_box_individual/distance_box_${v}.png`;
                case 'tsne':
                    const fTsne = ((fileRegistry[v] || {})[ckpt] || {}).folds?.[fold]?.tsne;
                    return fTsne || ((fileRegistry[v] || {})[ckpt] || {}).combined?.tsne || null;
                case 'pca':
                    return ((fileRegistry[v] || {})[ckpt] || {}).folds?.[fold]?.pca || null;
                case 'kde':
                    const fKde = ((fileRegistry[v] || {})[ckpt] || {}).folds?.[fold]?.kde;
                    return fKde || ((fileRegistry[v] || {})[ckpt] || {}).combined?.kde || null;
                case 'merged':
                    return ((fileRegistry[v] || {})[ckpt] || {}).combined?.merged || null;
                case 'confusion_matrix':
                    const cms = ((fileRegistry[v] || {})[ckpt] || {}).folds?.[fold]?.cms;
                    return (cms && cms.length > 0) ? cms[0].path : null;
                default:
                    return null;
            }
        }

        // Handle missing images
        function handleImageError(img) {
            const container = img.parentElement;
            container.onclick = null;
            container.style.cursor = 'default';
            container.innerHTML = `<div class="no-data-msg">Plot not applicable or data missing for this configuration</div>`;
        }

        // ────────────────────────────────────────────────────────
        // Render Plots dynamically
        // ────────────────────────────────────────────────────────
        function renderWorkspace() {
            const container = document.getElementById('plots-container');
            container.innerHTML = '';
            
            if (currentTab === 'overview') {
                updateSelectorVisibility();
                
                const checkedCbs = document.querySelectorAll('input[name="comp-exp"]:checked');
                const selectedExpKeys = Array.from(checkedCbs).map(cb => cb.value);
                
                if (selectedExpKeys.length === 0) {
                    container.innerHTML = '<div class="no-data-msg" style="grid-column: 1/-1;">Please select at least one experiment above to display comparisons.</div>';
                    return;
                }
                
                const plotType = document.getElementById('comp-plot-type').value;
                const ckpt = document.getElementById('comp-ckpt').value;
                const fold = parseInt(document.getElementById('comp-fold').value);
                
                const grid = document.createElement('div');
                grid.className = 'plots-grid';
                grid.style.display = 'grid';
                grid.style.gridTemplateColumns = `repeat(auto-fit, minmax(280px, 1fr))`;
                grid.style.gap = '20px';
                
                selectedExpKeys.forEach(v => {
                    const path = resolvePlotPath(v, plotType, ckpt, fold);
                    const label = lossLabels[v];
                    
                    let cardTitle = `${label}`;
                    if (plotType !== 'dist_box' && plotType !== 'merged') {
                        cardTitle += ` (${ckpt.replace('_net','').replace('best_','').toUpperCase()} | Fold ${fold})`;
                    }
                    
                    grid.appendChild(createPlotCard(cardTitle, path, true));
                });
                
                container.appendChild(grid);
            } else {
                const ckpt = document.getElementById('select-ckpt').value;
                const fold = parseInt(document.getElementById('select-fold').value);
                const reg = fileRegistry[currentTab][ckpt];
                
                const expSec = document.createElement('div');
                expSec.className = 'plots-grid';
                
                // --- Part A: Combined Plots (Aggregate folds) ---
                if (reg.combined.merged) {
                    expSec.appendChild(createPlotCard("Aggregate Fold Representation: Combined t-SNE & KDE Density", reg.combined.merged));
                } else {
                    if (reg.combined.tsne) {
                        expSec.appendChild(createPlotCard("Combined t-SNE Plot (All Folds)", reg.combined.tsne));
                    }
                    if (reg.combined.kde) {
                        expSec.appendChild(createPlotCard("Combined KDE Severity distribution (All Folds)", reg.combined.kde));
                    }
                }
                
                // --- Part B: Per-Fold Breakdown Plots ---
                const fData = reg.folds[fold] || {};
                if (fData.tsne) {
                    expSec.appendChild(createPlotCard(`Fold ${fold} - t-SNE Feature Projection`, fData.tsne));
                }
                if (fData.pca) {
                    expSec.appendChild(createPlotCard(`Fold ${fold} - PCA Feature Projection`, fData.pca));
                }
                if (fData.kde) {
                    expSec.appendChild(createPlotCard(`Fold ${fold} - KDE Latent Severity density`, fData.kde));
                }
                
                // --- Part C: Confusion Matrices (If present) ---
                if (fData.cms && fData.cms.length > 0) {
                    fData.cms.forEach(cm => {
                        const title = `Fold ${fold} - Confusion Matrix (${cm.train_mode === '3c' ? '3-Class labels training' : '4-Class labels training'} | ${ckpt.replace('_net','')})`;
                        expSec.appendChild(createPlotCard(title, cm.path));
                    });
                }
                
                // --- Part D: Comparative & Prototype Distance Plots ---
                const umapPath = `plots/umap_individual/umap_${currentTab}_${ckpt}_fold${fold}.png`;
                const scatterPath = `plots/distance_scatter_individual/distance_scatter_${currentTab}_fold${fold}.png`;
                const boxPath = `plots/distance_box_individual/distance_box_${currentTab}.png`;
                
                expSec.appendChild(createPlotCard(`Fold ${fold} - UMAP Feature Projection`, umapPath, true));
                expSec.appendChild(createPlotCard(`Fold ${fold} - Distance to CN vs AD Prototype Scatter`, scatterPath, true));
                expSec.appendChild(createPlotCard(`All Folds - Euclidean Distance to AD Prototype Boxplot`, boxPath, true));
                
                container.appendChild(expSec);
            }
        }
        
        function createPlotCard(title, relativePath, checkError = false) {
            const card = document.createElement('div');
            card.className = 'plot-card';
            
            const errHandler = checkError ? `onerror="handleImageError(this)"` : '';
            
            card.innerHTML = `
                <div class="plot-title">${title}</div>
                <div class="plot-img-container" onclick="openLightbox('${relativePath}')">
                    <img class="plot-img" src="${relativePath}" ${errHandler} alt="${title}">
                </div>
            `;
            return card;
        }

        function onFilterChange() {
            renderWorkspace();
        }

        // ────────────────────────────────────────────────────────
        // Lightbox Functions (Image Zoom)
        // ────────────────────────────────────────────────────────
        function openLightbox(src) {
            const lb = document.getElementById('lightbox');
            const lbImg = document.getElementById('lightbox-img');
            lbImg.src = src;
            lb.style.display = 'flex';
        }
        function closeLightbox() {
            document.getElementById('lightbox').style.display = 'none';
        }
        
        // Load initial state
        initComparisonCheckboxes();
        switchTab('overview');
    </script>
</body>
</html>
"""

# Inject data via .replace() to avoid f-string template brace conflicts
html_content = html_template.replace('__FILE_REGISTRY__', json.dumps(file_registry))
html_content = html_content.replace('__COMPARATIVE_REGISTRY__', json.dumps(comparative_registry))
html_content = html_content.replace('__METRICS_REGISTRY__', json.dumps(metrics_registry))
html_content = html_content.replace('__LOSS_LABELS__', json.dumps(LOSS_LABELS))
html_content = html_content.replace('__LOSS_VARIANTS__', json.dumps(LOSS_VARIANTS))

with open(os.path.join(dest_dir, 'index.html'), 'w') as f:
    f.write(html_content)

print(f"Unified dashboard compiled successfully at: {os.path.join(dest_dir, 'index.html')}")

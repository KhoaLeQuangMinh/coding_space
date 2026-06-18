import os

out_path = "/Users/khoale/Downloads/combined_latent_report.html"
tsne_dir = "analysis_output_tSNE/fold_plots"
pca_dir = "analysis_output_tSNE/plot_pca_folds"
kde_dir = "analysis_output_tSNE/plot_kde_folds"

VARIANTS = [
    'ce', 
    'ins2ins', 
    'ins2cls', 
    'full',
    'exclude_ins2ins',
    'exclude_ins2cls',
    'exp_triplet_ins2cls',
    'triplet_only',
    'hierarchical_triplet_only',
    'exp_hierarchical_triplet_ins2cls',
    'full_4class',
    'exp_triplet_ins2cls_4class',
    'hierarchical_triplet_only_4class',
    'qwk_hierarchical_triplet_4class',
    'exp_3pole_local',
    'exp_3pole_global'
]

CHECKPOINTS = ['best_2c_net', 'best_3c_net', 'best_4c_net']
FOLDS = [1, 2, 3, 4, 5]

variant_names = {
    'ce': 'L_CE',
    'ins2ins': 'L_CE + L_Ins2Ins',
    'ins2cls': 'L_CE + L_Ins2Ins + L_Ins2Cls',
    'full': 'L_CE + L_Ins2Ins + L_Ins2Cls + L_Cls2Cls',
    'exclude_ins2ins': 'L_CE + L_Ins2Cls + L_Cls2Cls',
    'exclude_ins2cls': 'L_CE + L_Ins2Ins + L_Cls2Cls',
    'exp_triplet_ins2cls': 'L_CE + L_Ins2Ins + L_Triplet + L_Cls2Cls',
    'triplet_only': 'L_CE + L_Triplet',
    'hierarchical_triplet_only': 'L_CE + L_Hierarchical_Triplet',
    'exp_hierarchical_triplet_ins2cls': 'L_CE + L_Ins2Ins + L_Hierarchical_Triplet + L_Cls2Cls',
    'full_4class': 'L_CE + L_Ins2Ins + L_Ins2Cls + L_Cls2Cls (4-Class)',
    'exp_triplet_ins2cls_4class': 'L_CE + L_Ins2Ins + L_Triplet + L_Cls2Cls (4-Class)',
    'hierarchical_triplet_only_4class': 'L_CE + L_Hierarchical_Triplet (4-Class)',
    'qwk_hierarchical_triplet_4class': 'L_QWK + L_Hierarchical_Triplet (4-Class)',
    'exp_3pole_local': 'L_CE + L_Ins2Ins + L_3Pole_Triplet (Local) + L_Cls2Cls',
    'exp_3pole_global': 'L_CE + L_Ins2Ins + L_3Pole_Triplet (Global) + L_Cls2Cls'
}

html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Combined PCA & t-SNE Latent Space Analysis</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0f111a;
            --surface: #1e2130;
            --primary: #6366f1;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --border-light: rgba(255,255,255,0.1);
        }
        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            margin: 0;
            padding: 40px;
        }
        header {
            text-align: center;
            margin-bottom: 50px;
        }
        h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            background: linear-gradient(90deg, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        p.subtitle {
            color: var(--text-muted);
            font-size: 1.1rem;
            max-width: 900px;
            margin: 0 auto;
            line-height: 1.6;
        }
        .analysis-card {
            background: rgba(30, 33, 48, 0.6);
            border: 1px solid var(--border-light);
            border-radius: 12px;
            padding: 20px 30px;
            margin-bottom: 40px;
            backdrop-filter: blur(10px);
        }
        .analysis-card h3 {
            color: #818cf8;
            margin-top: 0;
        }
        .variant-section {
            margin-bottom: 80px;
            border-top: 1px solid var(--border-light);
            padding-top: 40px;
        }
        .variant-title {
            font-size: 2rem;
            margin-bottom: 30px;
            color: #e2e8f0;
            text-align: center;
            background: rgba(255,255,255,0.05);
            padding: 15px;
            border-radius: 8px;
        }
        .checkpoint-block {
            margin-bottom: 50px;
            background: rgba(0,0,0,0.2);
            padding: 20px;
            border-radius: 12px;
            border: 1px solid var(--border-light);
        }
        .checkpoint-title {
            font-size: 1.4rem;
            color: #c084fc;
            margin-bottom: 20px;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 600;
        }
        .row-label {
            font-size: 1.1rem;
            font-weight: 600;
            color: #cbd5e1;
            margin-bottom: 10px;
            border-left: 4px solid #6366f1;
            padding-left: 10px;
        }
        .folds-container {
            display: flex;
            gap: 15px;
            overflow-x: auto;
            padding-bottom: 20px;
            margin-bottom: 20px;
        }
        .fold-card {
            flex: 0 0 19%;
            background: var(--surface);
            border-radius: 8px;
            padding: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            text-align: center;
            transition: transform 0.2s;
        }
        .fold-card:hover {
            transform: translateY(-5px);
        }
        .fold-card img {
            width: 100%;
            border-radius: 4px;
            background: white; 
        }
        .fold-label {
            margin-top: 10px;
            font-weight: 600;
            font-size: 0.9rem;
            color: #cbd5e1;
        }
    </style>
</head>
<body>

    <header>
        <h1>Latent Manifold: t-SNE vs PCA</h1>
        <p class="subtitle">Direct comparison between non-linear local projection (t-SNE) and linear global projection (PCA) across all 5 folds. This perfectly illustrates the "t-SNE Illusion" versus true geometric ordinality.</p>
    </header>

    <div class="analysis-card">
        <h3>🔍 Reading This Report</h3>
        <p><strong>Top Row (t-SNE):</strong> Shows local cluster separation. While beautiful, it heavily distorts global distance. Baseline models will often look falsely separated here.</p>
        <p><strong>Middle Row (PCA):</strong> Shows strict linear separation. This is the ultimate proof of manifold quality. If a model collapses into a single blob here, the linear classifier failed. If the model stretches into a continuous spectrum (like the Hierarchical variants), it successfully learned the biological disease progression.</p>
        <p><strong>Bottom Row (KDE):</strong> Shows the 1-dimensional distribution of Latent Severity (similarity to AD prototype minus similarity to CN prototype). A perfect model creates four distinct, sequentially ordered peaks without overlap.</p>
    </div>

"""

for variant in VARIANTS:
    v_name = variant_names.get(variant, variant)
    html += f'    <div class="variant-section">\n'
    html += f'        <div class="variant-title">{v_name}</div>\n'
    
    for ckpt in CHECKPOINTS:
        html += f'        <div class="checkpoint-block">\n'
        html += f'            <div class="checkpoint-title">Checkpoint: {ckpt}</div>\n'
        
        # t-SNE Row
        html += f'            <div class="row-label">t-SNE Projections</div>\n'
        html += f'            <div class="folds-container">\n'
        for fold in FOLDS:
            img_filename = f"tsne_{variant}_{ckpt}_fold{fold}.png"
            img_path = f"{tsne_dir}/{img_filename}"
            html += f'                <div class="fold-card">\n'
            html += f'                    <img src="{img_path}" alt="t-SNE Fold {fold}" loading="lazy">\n'
            html += f'                    <div class="fold-label">Fold {fold}</div>\n'
            html += f'                </div>\n'
        html += f'            </div>\n'
        
        # PCA Row
        html += f'            <div class="row-label">PCA Projections</div>\n'
        html += f'            <div class="folds-container">\n'
        for fold in FOLDS:
            img_filename = f"pca_{variant}_{ckpt}_fold{fold}.png"
            img_path = f"{pca_dir}/{img_filename}"
            html += f'                <div class="fold-card">\n'
            html += f'                    <img src="{img_path}" alt="PCA Fold {fold}" loading="lazy">\n'
            html += f'                    <div class="fold-label">Fold {fold}</div>\n'
            html += f'                </div>\n'
        html += f'            </div>\n'
        
        # KDE Row
        html += f'            <div class="row-label">1D Latent KDE Distributions</div>\n'
        html += f'            <div class="folds-container">\n'
        for fold in FOLDS:
            img_filename = f"latent_kde_{variant}_{ckpt}_fold{fold}.png"
            img_path = f"{kde_dir}/{img_filename}"
            html += f'                <div class="fold-card">\n'
            html += f'                    <img src="{img_path}" alt="KDE Fold {fold}" loading="lazy">\n'
            html += f'                    <div class="fold-label">Fold {fold}</div>\n'
            html += f'                </div>\n'
        html += f'            </div>\n'
        
        html += f'        </div>\n'
        
    html += f'    </div>\n'

html += """
</body>
</html>
"""

with open(out_path, "w") as f:
    f.write(html)

print(f"Combined report generated successfully at {out_path}")

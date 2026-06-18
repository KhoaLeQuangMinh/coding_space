import os
import glob

def generate_html():
    out_dir = '/Users/khoale/Downloads/ablation_result/proposed/'
    html_path = os.path.join(out_dir, 'index.html')
    
    plot_types = ['1D_PCA', 'UMAP', 'Distance_Scatter', 'Confusion_Matrix_Detailed_2c', 'Confusion_Matrix_Detailed_3c', 'Confusion_Matrix_Detailed_4c']
    networks = ['best_2c_net', 'best_3c_net', 'best_4c_net']
    folds = [1, 2, 3, 4, 5]
    
    html_content = """
    <html>
    <head>
        <title>Proposed Visuals Viewer</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f9f9f9; }
            h1 { text-align: center; color: #333; }
            h2 { color: #555; border-bottom: 2px solid #ccc; padding-bottom: 5px; margin-top: 40px;}
            h3 { color: #666; margin-top: 20px;}
            .img-container { text-align: center; margin-bottom: 30px; background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
            img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }
            .nav { position: sticky; top: 0; background: white; padding: 10px; border-bottom: 1px solid #ddd; z-index: 100; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
            .nav a { margin: 0 15px; text-decoration: none; color: #0066cc; font-weight: bold; }
            .nav a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <div class="nav">
            <a href="#1D_PCA">1D PCA Projections</a>
            <a href="#UMAP">2D UMAP Projections</a>
            <a href="#Distance_Scatter">Euclidean Distance Scatter</a>
            <a href="#Confusion_Matrix_Detailed_2c">2-Class Confusion Matrix Grids</a>
            <a href="#Confusion_Matrix_Detailed_3c">3-Class Confusion Matrix Grids</a>
            <a href="#Confusion_Matrix_Detailed_4c">4-Class Confusion Matrix Grids</a>
        </div>
        
        <h1>Advanced Visual Analysis (CE vs HOPE vs Proposed Triplet)</h1>
    """
    
    for p_type in plot_types:
        html_content += f'<h2 id="{p_type}">{p_type.replace("_", " ")}</h2>\n'
        for net in networks:
            if p_type.startswith('Confusion_Matrix_Detailed'):
                img_name = f"{p_type}_{net}.png"
                if os.path.exists(os.path.join(out_dir, img_name)):
                    html_content += f"""
                    <div class="img-container">
                        <h4>Network: {net} (6 Grids: Folds 1-5 + Aggregated)</h4>
                        <img src="{img_name}" alt="{img_name}">
                    </div>
                    """
            else:
                # Skip if the combination doesn't exist (e.g. Distance_Scatter only has 2c)
                files_exist = any(os.path.exists(os.path.join(out_dir, f"{p_type}_{net}_fold{f}.png")) for f in folds)
                if not files_exist:
                    continue
                    
                html_content += f'<h3>Network: {net}</h3>\n'
                for fold in folds:
                    img_name = f"{p_type}_{net}_fold{fold}.png"
                    if os.path.exists(os.path.join(out_dir, img_name)):
                        html_content += f"""
                        <div class="img-container">
                            <h4>Fold {fold}</h4>
                            <img src="{img_name}" alt="{img_name}">
                        </div>
                        """
                    
    html_content += """
    </body>
    </html>
    """
    
    with open(html_path, 'w') as f:
        f.write(html_content)
        
    print(f"HTML Viewer generated at: {html_path}")

if __name__ == '__main__':
    generate_html()

import re
import os
import PyPDF2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def parse_matrices(text):
    # results[variant][model][size][fold] = matrix
    results = {'CE': {}, 'FULL': {}, 'TRIPLET_ONLY': {}, 'EXP_3POLE_LOCAL': {}, 'EXP_3POLE_GLOBAL': {}, '3POLE_LOCAL_ONLY': {}, '3POLE_GLOBAL_ONLY': {}}
    for v in results:
        for m in ['best_2c_net', 'best_3c_net', 'best_4c_net']:
            results[v][m] = {'3c': {}, '4c': {}}
            
    sections = re.split(r'TESTING (?:LOSS|EMA) ABLATION:\s*', text)
    
    for section in sections[1:]:
        header_text = section[:500]
        v_match = re.search(r'(CE|FULL|TRIPLET_ONLY)', header_text)
        m_match = re.search(r'(best_[234]c_net)', header_text)
        f_match = re.search(r'FOLD\(S\):\s*(\d+)', header_text)
        
        if not v_match or not m_match or not f_match:
            continue
            
        variant = v_match.group(1)
        model = m_match.group(1)
        fold = int(f_match.group(1))
        
        # Extract 4-class block
        block_4c_match = re.search(r'Combined 4-Class Confusion Matrix.*?(?=Combined 3-Class Confusion Matrix|\Z)', section, re.DOTALL)
        if block_4c_match:
            block_4c = block_4c_match.group(0)
            rows_4c = []
            for c in ['True CN', 'True sMCI', 'True pMCI', 'True AD']:
                m_row = re.search(c + r'\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|', block_4c)
                if m_row:
                    rows_4c.append([int(x) for x in m_row.groups()])
            if len(rows_4c) == 4:
                results[variant][model]['4c'][fold] = np.array(rows_4c)

        # Extract 3-class block
        block_3c_match = re.search(r'Combined 3-Class Confusion Matrix.*?(?=\Z)', section, re.DOTALL)
        if block_3c_match:
            block_3c = block_3c_match.group(0)
            rows_3c = []
            for c in ['True CN', 'True MCI', 'True AD']:
                m_row = re.search(c + r'\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|', block_3c)
                if m_row:
                    rows_3c.append([int(x) for x in m_row.groups()])
            if len(rows_3c) == 3:
                results[variant][model]['3c'][fold] = np.array(rows_3c)
                
    return results

def parse_txt_logs(results):
    log_dir = '/Users/khoale/Downloads/ablation_result/working/coding_space/Hope Implementation/checkpoints'
    for v_display, v_folder in [
        ('EXP_3POLE_LOCAL', 'ablation_loss_exp_3pole_local'), 
        ('EXP_3POLE_GLOBAL', 'ablation_loss_exp_3pole_global'),
        ('3POLE_LOCAL_ONLY', 'ablation_loss_3pole_local_only'),
        ('3POLE_GLOBAL_ONLY', 'ablation_loss_3pole_global_only')
    ]:
        for model in ['best_2c_net', 'best_3c_net', 'best_4c_net']:
            net_short = model.split('_')[1] # '2c', '3c', '4c'
            for fold in [1, 2, 3, 4, 5]:
                log_path = os.path.join(log_dir, f"{v_folder}_fold{fold}", f"{v_folder}_test_log_best_{net_short}.txt")
                if not os.path.exists(log_path):
                    continue
                
                with open(log_path, 'r') as f:
                    text = f.read()
                    
                # 4-class
                block_4c_match = re.search(r'COMBINED 4-CLASS CONFUSION MATRIX.*?(?=COMBINED 3-CLASS|\Z)', text, re.DOTALL)
                if block_4c_match:
                    block_4c = block_4c_match.group(0)
                    rows_4c = []
                    for c in ['True CN', 'True sMCI', 'True pMCI', 'True AD']:
                        m_row = re.search(c + r'\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|', block_4c)
                        if m_row:
                            rows_4c.append([int(x) for x in m_row.groups()])
                    if len(rows_4c) == 4:
                        results[v_display][model]['4c'][fold] = np.array(rows_4c)

                # 3-class
                block_3c_match = re.search(r'COMBINED 3-CLASS CONFUSION MATRIX.*?(?=\Z)', text, re.DOTALL)
                if block_3c_match:
                    block_3c = block_3c_match.group(0)
                    rows_3c = []
                    for c in ['True CN', 'True MCI', 'True AD']:
                        m_row = re.search(c + r'\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|', block_3c)
                        if m_row:
                            rows_3c.append([int(x) for x in m_row.groups()])
                    if len(rows_3c) == 3:
                        results[v_display][model]['3c'][fold] = np.array(rows_3c)
    return results

def compute_2c_from_csvs(results):
    csv_dir = '/Users/khoale/Downloads/analysis_output_tSNE/extracted_features'
    variants = {
        'CE': 'ce', 
        'FULL': 'full', 
        'TRIPLET_ONLY': 'triplet_only', 
        'EXP_3POLE_LOCAL': 'exp_3pole_local', 
        'EXP_3POLE_GLOBAL': 'exp_3pole_global',
        '3POLE_LOCAL_ONLY': '3pole_local_only',
        '3POLE_GLOBAL_ONLY': '3pole_global_only'
    }
    networks = ['best_2c_net', 'best_3c_net', 'best_4c_net']
    folds = [1, 2, 3, 4, 5]
    
    for v_display, v_file in variants.items():
        for net in networks:
            results[v_display][net]['2c'] = {}
            for fold in folds:
                cm = np.zeros((2,2), dtype=int)
                csv_path = os.path.join(csv_dir, f"{v_file}_{net}_fold{fold}.csv")
                if not os.path.exists(csv_path):
                    continue
                try:
                    df = pd.read_csv(csv_path, header=0) # first row is header
                    severities = df.iloc[:, 0].astype(float).values
                    labels = df.iloc[:, 1].astype(str).values
                    
                    for sev, label in zip(severities, labels):
                        if label == 'sMCI':
                            pred = 1 if sev > 0 else 0
                            cm[0, pred] += 1
                        elif label == 'pMCI':
                            pred = 1 if sev > 0 else 0
                            cm[1, pred] += 1
                            
                    if np.sum(cm) > 0:
                        results[v_display][net]['2c'][fold] = cm
                except Exception as e:
                    print(f"Error processing 2c CSV {csv_path}: {e}")
                    pass
            
    return results

def plot_confusion_matrices(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    var_map = {
        'CE': 'CE (Baseline)',
        'FULL': 'HOPE (Full)',
        'TRIPLET_ONLY': 'Proposed (Triplet Only)',
        'EXP_3POLE_LOCAL': '3-Pole Triplet (Local)',
        'EXP_3POLE_GLOBAL': '3-Pole Triplet (Global)',
        '3POLE_LOCAL_ONLY': '3-Pole Local Only',
        '3POLE_GLOBAL_ONLY': '3-Pole Global Only'
    }
    
    networks = ['best_2c_net', 'best_3c_net', 'best_4c_net']
    folds_disp = [1, 2, 3, 4, 5, 'Aggregated']
    
    for net in networks:
        for size in ['2c', '3c', '4c']:
            fig, axes = plt.subplots(7, 6, figsize=(32, 38))
            if size == '2c':
                title_size = '2-Class (sMCI vs pMCI)'
                labels = ['sMCI', 'pMCI']
                cmap = 'Oranges'
            elif size == '3c':
                title_size = '3-Class (CN, MCI, AD)'
                labels = ['CN', 'MCI', 'AD']
                cmap = 'Greens'
            else:
                title_size = '4-Class (CN, sMCI, pMCI, AD)'
                labels = ['CN', 'sMCI', 'pMCI', 'AD']
                cmap = 'Blues'
                
            fig.suptitle(f'{title_size} Confusion Matrices across Folds - {net}', fontsize=24)
            
            for r, variant in enumerate(['CE', 'FULL', 'TRIPLET_ONLY', 'EXP_3POLE_LOCAL', 'EXP_3POLE_GLOBAL', '3POLE_LOCAL_ONLY', '3POLE_GLOBAL_ONLY']):
                var_dict = results[variant][net][size]
                
                # Calculate aggregated matrix
                agg_mat = None
                valid_mats = [mat for f, mat in var_dict.items() if isinstance(f, int)]
                if valid_mats:
                    agg_mat = sum(valid_mats)
                
                for c, fold in enumerate(folds_disp):
                    ax = axes[r, c]
                    
                    if fold == 'Aggregated':
                        cm = agg_mat
                        col_title = f"{var_map[variant]}\nCombined All Folds"
                    else:
                        cm = var_dict.get(fold, None)
                        col_title = f"{var_map[variant]}\nFold {fold}"
                        
                    if cm is not None:
                        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, 
                                    xticklabels=labels, yticklabels=labels, ax=ax, annot_kws={"size": 16})
                        ax.set_title(col_title, fontsize=16)
                        if c == 0:
                            ax.set_ylabel('True Label', fontsize=14)
                        if r == 6:
                            ax.set_xlabel('Predicted Label', fontsize=14)
                    else:
                        ax.set_title(f"{col_title}\n(Data Missing)", fontsize=16)
                        ax.axis('off')
                        
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            out_path = os.path.join(out_dir, f"Confusion_Matrix_Detailed_{size}_{net}.png")
            plt.savefig(out_path, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"Saved: {out_path}")

def main():
    pdfs = [
        '/Users/khoale/Downloads/Lasthope1.pdf',
        '/Users/khoale/Downloads/Lasthope2.pdf',
        '/Users/khoale/Downloads/CE + Triplets.pdf'
    ]
    
    full_text = ""
    for pdf in pdfs:
        print(f"Extracting {pdf}...")
        full_text += extract_text_from_pdf(pdf) + "\n"
        
    print("Parsing matrices...")
    results = parse_matrices(full_text)
    results = parse_txt_logs(results)
    results = compute_2c_from_csvs(results)
    
    print("Plotting detailed grids...")
    out_dir = '/Users/khoale/Downloads/ablation_result/proposed/'
    plot_confusion_matrices(results, out_dir)

if __name__ == '__main__':
    main()

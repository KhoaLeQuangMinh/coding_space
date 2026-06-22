import os
import pandas as pd
import numpy as np

results_dir = '/Users/khoale/Downloads/ablation_result/working/coding_space/Hope Implementation/checkpoints'
save_dir = '/Users/khoale/Desktop/Coding_Space/Hope Implementation'

variants = [
    'ce',
    'ins2ins',
    'ins2cls',
    'full',
    'exclude_ins2ins',
    'exclude_ins2cls',
    'exp_triplet_ins2cls',
    'triplet_only',
    'exp_3pole_local',
    'exp_3pole_global',
    '3pole_local_only',
    '3pole_global_only'
]

row_labels = {
    'ce': 'L_CE',
    'ins2ins': '+ L_Ins2Ins',
    'ins2cls': '+ L_Ins2Cls',
    'full': '+ L_Cls2Cls (HOPE)',
    'exclude_ins2ins': 'Exclude Ins2Ins Ablation',
    'exclude_ins2cls': 'Exclude Ins2Cls Ablation',
    'exp_triplet_ins2cls': 'Triplet Ins2Cls (Poles)',
    'triplet_only': 'CE + Triplet Only',
    'exp_3pole_local': 'L_CE + L_Ins2Ins + L_3Pole_Triplet (Local) + L_Cls2Cls',
    'exp_3pole_global': 'L_CE + L_Ins2Ins + L_3Pole_Triplet (Global) + L_Cls2Cls',
    '3pole_local_only': 'L_CE + L_3Pole_Triplet (Local) Only',
    '3pole_global_only': 'L_CE + L_3Pole_Triplet (Global) Only'
}

def export_table(target):
    rows = []
    
    for v in variants:
        metrics = {'Acc': [], 'AUC': [], 'F1': [], 'Prec': [], 'Recall': []}
        folds = 0
        
        for fold in range(1, 6):
            path = os.path.join(results_dir, f'ablation_loss_{v}_fold{fold}', 'test_metrics_best_2c.csv')
            if os.path.exists(path):
                df = pd.read_csv(path)
                folds += 1
                
                metrics['Acc'].append(df[f'Acc {target}-class'].iloc[0] * 100)
                metrics['AUC'].append(df[f'AUC {target}-class'].iloc[0] * 100)
                metrics['F1'].append(df[f'F1 {target}-class'].iloc[0] * 100)
                metrics['Prec'].append(df[f'Prec {target}-class'].iloc[0] * 100)
                metrics['Recall'].append(df[f'Recall {target}-class'].iloc[0] * 100)
                
        if folds > 0:
            m_acc = np.mean(metrics['Acc'])
            s_acc = np.std(metrics['Acc'], ddof=1)
            
            m_auc = np.mean(metrics['AUC'])
            s_auc = np.std(metrics['AUC'], ddof=1)
            
            m_f1 = np.mean(metrics['F1'])
            s_f1 = np.std(metrics['F1'], ddof=1)
            
            m_pre = np.mean(metrics['Prec'])
            s_pre = np.std(metrics['Prec'], ddof=1)
            
            m_rec = np.mean(metrics['Recall'])
            s_rec = np.std(metrics['Recall'], ddof=1)
            
            acc_str = f"{m_acc:.1f} ± {s_acc:.1f}"
            auc_str = f"{m_auc:.1f} ± {s_auc:.1f}"
            f1_str = f"{m_f1:.1f} ± {s_f1:.1f}"
            pre_str = f"{m_pre:.1f} ± {s_pre:.1f}"
            rec_str = f"{m_rec:.1f} ± {s_rec:.1f}"
            
            rows.append({
                'Variant': row_labels[v],
                'Model': 'best_2c_net',
                'N_Folds': folds,
                'ACC': acc_str,
                'AUC': auc_str,
                'F1-score': f1_str,
                'Precision': pre_str,
                'Recall': rec_str
            })
        else:
            rows.append({
                'Variant': row_labels[v],
                'Model': 'best_2c_net',
                'N_Folds': 0,
                'ACC': 'N/A',
                'AUC': 'N/A',
                'F1-score': 'N/A',
                'Precision': 'N/A',
                'Recall': 'N/A'
            })
            
    df_out = pd.DataFrame(rows)
    out_path = os.path.join(save_dir, f'table3_loss_ablation_{target}class.csv')
    df_out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

export_table(3)
export_table(4)

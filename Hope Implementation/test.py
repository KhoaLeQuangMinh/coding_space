import torch
import torch.nn as nn

from options.test_options import TestOptions
from utils.Dataset import *
from utils.test_data import *
from utils.tools import *
from utils.train_data import *

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import os

def run_test(opt, current_fold):
    model = define_Cls(opt.cls_type, class_num=opt.class_num, init_type=opt.init_type, init_gain=opt.init_gain, m=opt.m,
                       gpu_ids=opt.gpu_ids)

    # criterion preparation
    criterion = nn.CrossEntropyLoss()

    # dataset preparation
    test_dataset = Dataset(mode="test", data_dir=opt.data_dir, seed=opt.seed, kfold=opt.kfold, current_fold=current_fold)

    # test loader
    num_workers_test = max(0, int(opt.workers / 2))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=num_workers_test, pin_memory=True)

    # model loading
    if opt.kfold == 1:
        load_dir = opt.load_dir
    else:
        load_dir = f"{opt.load_dir}_fold{current_fold}/best_{opt.test_target}_net.pth"
    try:
        state_dict = torch.load(load_dir, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        # ema prototype
        model.prototypes = state_dict['prototypes'].cuda()
        model.cuda()
        print(f"loading weights from {load_dir}")
        print("Testing on the testing set")
        return test_data(model, test_loader, criterion)
    except FileNotFoundError:
        print(f"Weights {load_dir} not found. Ensure the model has been trained.")
        return None

if __name__ == '__main__':
    # -----  Loading the init options -----
    opt = TestOptions().parse()
    
    all_metrics = []
    
    if opt.specific_fold != -1:
        print(f"\n{'='*40}\nTesting SPECIFIC Fold {opt.specific_fold}/{opt.kfold} (Distributed Mode)\n{'='*40}\n")
        res = run_test(opt, opt.specific_fold)
        if res is not None:
            all_metrics.append(res)
    elif opt.kfold > 1:
        for f in range(1, opt.kfold + 1):
            print(f"\n{'='*40}\nTesting Fold {f}/{opt.kfold}\n{'='*40}\n")
            res = run_test(opt, f)
            if res is not None:
                all_metrics.append(res)
    else:
        res = run_test(opt, 1)
        if res is not None:
            all_metrics.append(res)

    if len(all_metrics) > 0:
        print(f"\n{'='*40}\nTesting Summary\n{'='*40}\n")
        
        summary_data = []
        for m in all_metrics:
            summary_data.append({
                'Loss': m['val_loss'],
                'Acc 4-class': m['val_acc_4class'],
                'F1 4-class': m['val_f1_4class'],
                'Acc 3-class': m['val_acc_3class'],
                'F1 3-class': m['val_f1_3class'],
                'MCI Acc': m['val_acc'],
                'MCI F1': m['val_f1_score'],
                'MCI SPE': m['val_spe'],
                'MCI SEN': m['val_sen'],
                'MCI AUC': m['val_auc'],
                'MCI Prec': m['val_precision']
            })
        
        df = pd.DataFrame(summary_data)
        df.index = [f"Fold {i+1}" for i in range(len(all_metrics))]
        
        if len(all_metrics) > 1:
            df.loc['Mean'] = df.mean()
            df.loc['Std'] = df.std()
        
        print(df.to_markdown())
        
        try:
            if opt.specific_fold != -1:
                expr_dir = os.path.join(opt.checkpoints_dir, f"{opt.name}_fold{opt.specific_fold}")
            else:
                expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
            os.makedirs(expr_dir, exist_ok=True)
            csv_path = os.path.join(expr_dir, f"test_metrics_best_{opt.test_target}.csv")
            df.to_csv(csv_path)
            print(f"\nSaved test metrics to {csv_path}")
        except Exception as e:
            print(f"Could not save test metrics csv: {e}")
        # Confusion Matrix for 4-class across all folds
        y_true_all = []
        y_pred_all = []
        for m in all_metrics:
            y_true_all.extend(m['y_true_4c'])
            y_pred_all.extend(m['y_pred_4c'])
        
        cm = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1, 2, 3])
        
        print("\nCombined 4-Class Confusion Matrix (CN=0, sMCI=1, pMCI=2, AD=3):")
        cm_df = pd.DataFrame(cm, index=['True CN', 'True sMCI', 'True pMCI', 'True AD'],
                             columns=['Pred CN', 'Pred sMCI', 'Pred pMCI', 'Pred AD'])
        print(cm_df.to_markdown())
        
        # Confusion Matrix for 3-class across all folds
        y_true_3c_all = []
        y_pred_3c_all = []
        for m in all_metrics:
            y_true_3c_all.extend(m['y_true_3c'])
            y_pred_3c_all.extend(m['y_pred_3c'])
            
        cm_3c = confusion_matrix(y_true_3c_all, y_pred_3c_all, labels=[0, 1, 2])
        print("\nCombined 3-Class Confusion Matrix (CN=0, MCI=1, AD=2):")
        cm_3c_df = pd.DataFrame(cm_3c, index=['True CN', 'True MCI', 'True AD'],
                             columns=['Pred CN', 'Pred MCI', 'Pred AD'])
        print(cm_3c_df.to_markdown())
        
        # Save visual CMs
        try:
            if opt.specific_fold != -1:
                expr_dir = os.path.join(opt.checkpoints_dir, f"{opt.name}_fold{opt.specific_fold}")
            else:
                expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
            os.makedirs(expr_dir, exist_ok=True)
            
            # Plot 4-class
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
            plt.title('Combined 4-Class Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            cm_path = os.path.join(expr_dir, f"{opt.name}_confusion_matrix_4c_best_{opt.test_target}.png")
            plt.savefig(cm_path)
            
            # Plot 3-class
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm_3c_df, annot=True, fmt='d', cmap='Greens')
            plt.title('Combined 3-Class Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            cm_path_3c = os.path.join(expr_dir, f"{opt.name}_confusion_matrix_3c_best_{opt.test_target}.png")
            plt.savefig(cm_path_3c)
            
            print(f"\nSaved Confusion Matrix plots to {expr_dir}")
        except Exception as e:
            print(f"Could not save confusion matrix plot: {e}")

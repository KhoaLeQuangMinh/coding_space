import time

import numpy as np
import torch
from sklearn.metrics import f1_score, recall_score, roc_auc_score, accuracy_score, precision_score, cohen_kappa_score
from tqdm.auto import tqdm


def test_data(model, test_dataloaders, criterion):
    '''
    test process
    :param model: the corresponding model
    :param test_dataloaders: the smci/pmci test loader
    :param criterion: CE criterion
    :return:
    '''
    start = time.time()
    val_loss = 0.
    val_samples = 0
    y_val_true = []
    y_val_pred = []
    val_prob_all = []
    
    y_val_true_4class = []
    y_val_pred_4class = []
    val_prob_4class = []
    
    y_val_true_3class = []
    y_val_pred_3class = []
    val_prob_3class = []
    
    with torch.no_grad():
        model.eval()
        for ii, batch in enumerate(test_dataloaders):
            images, labels = batch[0].cuda(), batch[1].cuda()
            labels_4c = batch[2].cuda() if len(batch) > 2 else None
            
            num_classes = model.module.num_classes if hasattr(model, 'module') else model.num_classes
            if num_classes == 4 and labels_4c is not None:
                labels = labels_4c

            _, x, outputs = model(images)
            
            _, val_predicted = torch.max(outputs.data, 1)
            _, x_predicted = torch.max(x.data, 1)
            
            for b, s, t in zip(x_predicted, val_predicted, labels):
                t_val = t.item()
                if num_classes == 3:
                    # 4-class mapping
                    if b == 0:
                        pred_4c = 0
                    elif b == 2:
                        pred_4c = 3
                    else:
                        pred_4c = 1 if s == 0 else 2
                    
                    # 3-class mapping
                    pred_3c = b.item()
                    true_3c = 0 if t_val == 0 else (1 if t_val in [1, 2] else 2)
                else: # num_classes == 4
                    pred_4c = b.item()
                    pred_3c = 0 if pred_4c == 0 else (1 if pred_4c in [1, 2] else 2)
                    true_3c = 0 if t_val == 0 else (1 if t_val in [1, 2] else 2)
                
                y_val_pred_4class.append(pred_4c)
                y_val_true_4class.append(t_val)
                y_val_pred_3class.append(pred_3c)
                y_val_true_3class.append(true_3c)
            
            # Continuous probability extraction for AUC
            prob_bin = outputs.softmax(dim=-1)
            
            if num_classes == 3:
                prob_3c = x.softmax(dim=-1)
                prob_4c = torch.zeros(images.size(0), 4, device=x.device)
                prob_4c[:, 0] = prob_3c[:, 0]
                prob_4c[:, 1] = prob_3c[:, 1] * prob_bin[:, 0]
                prob_4c[:, 2] = prob_3c[:, 1] * prob_bin[:, 1]
                prob_4c[:, 3] = prob_3c[:, 2]
            else:
                prob_4c = x.softmax(dim=-1)
                prob_3c = torch.zeros(images.size(0), 3, device=x.device)
                prob_3c[:, 0] = prob_4c[:, 0]
                prob_3c[:, 1] = prob_4c[:, 1] + prob_4c[:, 2]
                prob_3c[:, 2] = prob_4c[:, 3]
            
            val_prob_3class.extend(prob_3c.cpu().detach().numpy())
            val_prob_4class.extend(prob_4c.cpu().detach().numpy())

            mci_mask = (labels == 1) | (labels == 2)
            if mci_mask.any():
                if num_classes == 4:
                    mci_outputs = x[mci_mask][:, 1:3]  # Extract logits for sMCI (1) and pMCI (2)
                else:
                    mci_outputs = outputs[mci_mask]
                mci_labels = labels[mci_mask] - 1  # 1->0, 2->1
                loss = criterion(mci_outputs, mci_labels)
                val_loss += loss.item() * mci_labels.size(0)
                val_samples += mci_labels.size(0)
                
                _, mci_predicted = torch.max(mci_outputs.data, 1)
                y_val_true.extend(np.ravel(np.squeeze(mci_labels.cpu().detach().numpy())).tolist())
                y_val_pred.extend(np.ravel(np.squeeze(mci_predicted.cpu().detach().numpy())).tolist())
                mci_outputs_prob = mci_outputs.softmax(dim=-1)
                val_prob_all.extend(mci_outputs_prob[:, 1].cpu().detach().numpy())

    # loss logging
    val_loss = val_loss / max(1, val_samples)
    
    if len(y_val_true_4class) > 0:
        val_acc_4class = accuracy_score(y_val_true_4class, y_val_pred_4class)
        val_f1_4class = f1_score(y_val_true_4class, y_val_pred_4class, average='weighted')
        val_prec_4class = precision_score(y_val_true_4class, y_val_pred_4class, average='weighted', zero_division=0)
        val_rec_4class = recall_score(y_val_true_4class, y_val_pred_4class, average='weighted')
        try:
            val_auc_4class = roc_auc_score(y_val_true_4class, np.array(val_prob_4class), multi_class='ovr', average='weighted')
        except ValueError:
            val_auc_4class = 0.0
            
        val_qwk_4class = cohen_kappa_score(y_val_true_4class, y_val_pred_4class, weights='quadratic')
        
        val_acc_3class = accuracy_score(y_val_true_3class, y_val_pred_3class)
        val_f1_3class = f1_score(y_val_true_3class, y_val_pred_3class, average='weighted')
        val_prec_3class = precision_score(y_val_true_3class, y_val_pred_3class, average='weighted', zero_division=0)
        val_rec_3class = recall_score(y_val_true_3class, y_val_pred_3class, average='weighted')
        try:
            val_auc_3class = roc_auc_score(y_val_true_3class, np.array(val_prob_3class), multi_class='ovr', average='weighted')
        except ValueError:
            val_auc_3class = 0.0
            
        val_qwk_3class = cohen_kappa_score(y_val_true_3class, y_val_pred_3class, weights='quadratic')
    else:
        val_acc_4class = val_f1_4class = val_prec_4class = val_rec_4class = val_auc_4class = val_qwk_4class = 0.0
        val_acc_3class = val_f1_3class = val_prec_3class = val_rec_3class = val_auc_3class = val_qwk_3class = 0.0

    # In case there are no MCI samples
    if len(y_val_true) > 0:
        val_acc = accuracy_score(y_val_true, y_val_pred)
        val_f1_score = f1_score(y_val_true, y_val_pred, average='weighted')
        val_recall = recall_score(y_val_true, y_val_pred, average='weighted')
        try:
            val_spe = recall_score(y_val_true, y_val_pred, pos_label=0, average='binary')
        except ValueError:
            val_spe = 0.0
        val_precision = precision_score(y_val_true, y_val_pred, average='weighted', zero_division=0)
        try:
            val_auc = roc_auc_score(y_val_true, val_prob_all, average='weighted')
        except ValueError:
            val_auc = 0.0
    else:
        val_acc = val_f1_score = val_recall = val_spe = val_precision = val_auc = 0.0

    print(
        'Test Loss:{:.3f}...'.format(val_loss),
        'Test Acc 4-class:{:.3f}...'.format(val_acc_4class),
        'Test QWK 4-class:{:.3f}...'.format(val_qwk_4class),
        'Test F1 4-class:{:.3f}...'.format(val_f1_4class),
        'Test Acc 3-class:{:.3f}...'.format(val_acc_3class),
        'Test QWK 3-class:{:.3f}...'.format(val_qwk_3class),
        'Test F1 3-class:{:.3f}...'.format(val_f1_3class),
        'Test Accuracy:{:.3f}...'.format(val_acc),
        'Test F1 Score:{:.3f}'.format(val_f1_score),
        'Test SPE:{:.3f}...'.format(val_spe),
        'Test SEN:{:.3f}...'.format(val_recall),
        'Test AUC:{:.3f}...'.format(val_auc),
        "Test precision:{:.3f}...".format(val_precision)
    )

    end = time.time()
    runing_time = end - start
    print('Testing time is {:.0f}m {:.0f}s'.format(runing_time // 60, runing_time % 60))

    metrics = {
        'val_loss': val_loss,
        'val_acc_4class': val_acc_4class,
        'val_qwk_4class': val_qwk_4class,
        'val_f1_4class': val_f1_4class,
        'val_prec_4class': val_prec_4class,
        'val_rec_4class': val_rec_4class,
        'val_auc_4class': val_auc_4class,
        'val_acc_3class': val_acc_3class,
        'val_qwk_3class': val_qwk_3class,
        'val_f1_3class': val_f1_3class,
        'val_prec_3class': val_prec_3class,
        'val_rec_3class': val_rec_3class,
        'val_auc_3class': val_auc_3class,
        'val_acc': val_acc,
        'val_f1_score': val_f1_score,
        'val_spe': val_spe,
        'val_sen': val_recall,
        'val_auc': val_auc,
        'val_precision': val_precision,
        'y_true_4c': y_val_true_4class,
        'y_pred_4c': y_val_pred_4class,
        'y_true_3c': y_val_true_3class,
        'y_pred_3c': y_val_pred_3class,
        'y_true_mci': y_val_true,
        'y_pred_mci': y_val_pred
    }
    return metrics

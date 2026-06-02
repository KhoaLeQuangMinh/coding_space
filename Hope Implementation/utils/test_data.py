import time

import numpy as np
import torch
from sklearn.metrics import f1_score, recall_score, roc_auc_score, accuracy_score, precision_score
from tqdm import tqdm


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
    
    with torch.no_grad():
        model.eval()
        for ii, (images, labels) in enumerate(tqdm(test_dataloaders)):
            images, labels = images.cuda(), labels.cuda()
            _, x, outputs = model(images)
            
            _, val_predicted = torch.max(outputs.data, 1)
            _, x_predicted = torch.max(x.data, 1)
            
            for b, s, t in zip(x_predicted, val_predicted, labels):
                if b == 0:
                    pred_4c = 0
                elif b == 2:
                    pred_4c = 3
                else:
                    pred_4c = 1 if s == 0 else 2
                
                y_val_pred_4class.append(pred_4c)
                y_val_true_4class.append(t.item())

            mci_mask = (labels == 1) | (labels == 2)
            if mci_mask.any():
                mci_outputs = outputs[mci_mask]
                mci_labels = labels[mci_mask] - 1  # 1->0, 2->1
                loss = criterion(mci_outputs, mci_labels)
                val_loss += loss.item() * mci_labels.size(0)
                val_samples += mci_labels.size(0)
                
                _, mci_predicted = torch.max(mci_outputs.data, 1)
                y_val_true.extend(np.ravel(np.squeeze(mci_labels.cpu().detach().numpy())).tolist())
                y_val_pred.extend(np.ravel(np.squeeze(mci_predicted.cpu().detach().numpy())).tolist())
                mci_outputs = mci_outputs.softmax(dim=-1)
                val_prob_all.extend(mci_outputs[:, 1].cpu().detach().numpy())

    # loss logging
    val_loss = val_loss / max(1, val_samples)
    
    val_acc_4class = accuracy_score(y_val_true_4class, y_val_pred_4class)
    val_f1_4class = f1_score(y_val_true_4class, y_val_pred_4class, average='weighted')
    
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
        'Val Loss:{:.3f}...'.format(val_loss),
        'Val Acc 4-class:{:.3f}...'.format(val_acc_4class),
        'Val F1 4-class:{:.3f}...'.format(val_f1_4class),
        'Val Accuracy:{:.3f}...'.format(val_acc),
        'Val F1 Score:{:.3f}'.format(val_f1_score),
        'Val SPE:{:.3f}...'.format(val_spe),
        'Val SEN:{:.3f}...'.format(val_recall),
        'Val AUC:{:.3f}...'.format(val_auc),
        "Val precision:{:.3f}...".format(val_precision)
    )

    end = time.time()
    runing_time = end - start
    print('Testing time is {:.0f}m {:.0f}s'.format(runing_time // 60, runing_time % 60))

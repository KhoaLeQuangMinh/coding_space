import csv
import os
import time

import numpy as np
import torch
import wandb
from sklearn.metrics import f1_score, recall_score, roc_auc_score, accuracy_score, precision_score, cohen_kappa_score
from tqdm.auto import tqdm
from models.qwk_loss import DifferentiableQWKLoss


def train_data(model, total_cn_loader, total_ad_loader, total_mci_loader,
               valid_dataloaders, epochs, optimizer, scheduler,
               basiccomputing, criterion, criterionRank,
               expr_dir, print_freq, save_epoch_freq, ablation_loss='full'
               ):
    '''
    train process
    :param model: the corresponding model
    :param total_cn_loader: the cn train loader
    :param total_ad_loader: the ad train loader
    :param total_mci_loader: the mci train loader
    :param valid_dataloaders: the smci/pmci valid loader
    :param epochs: epochs
    :param optimizer: the optimizer
    :param scheduler: the scheduler
    :param basiccomputing: basic computing component
    :param criterion: CE criterion
    :param criterionRank: Rank criterion
    :param expr_dir: the saved directory
    :param print_freq: print frequency
    :param save_epoch_freq: saving frequency
    :param ablation_loss: determines which loss components to use ('ce', 'ins2ins', 'ins2cls', 'full')
    :return:
    '''
    start = time.time()
    steps = 0
    history = []
    
    # Initialize Triple-Saving trackers
    best_val_acc_2c = 0.0
    best_val_acc_3c = 0.0
    best_val_acc_4c = 0.0
    qwk_loss_fn = None
    for e in tqdm(range(1, epochs + 1)):
        model.train()
        train_loss = 0.
        train_loss_CE = 0.
        train_loss_ins2ins = 0.
        train_loss_ins2cls = 0.
        train_loss_cls2cls = 0.
        train_loss_hyb = 0.
        y_train_true = []
        y_train_pred = []
        cn_iterator = iter(total_cn_loader)
        mci_iterator = iter(total_mci_loader)
        for ii, batch_ad in enumerate(total_ad_loader):
            imgs_ad = batch_ad[0]
            labels_ad = batch_ad[1]
            labels_ad_4c = batch_ad[2] if len(batch_ad) > 2 else None
            
            steps += 1
            try:
                batch_cn = next(cn_iterator)
                imgs_cn, labels_cn = batch_cn[0], batch_cn[1]
                labels_cn_4c = batch_cn[2] if len(batch_cn) > 2 else None
                
                batch_mci = next(mci_iterator)
                imgs_mci, labels_mci = batch_mci[0], batch_mci[1]
                labels_mci_4c = batch_mci[2] if len(batch_mci) > 2 else None
            except StopIteration:
                cn_iterator = iter(total_cn_loader)
                mci_iterator = iter(total_mci_loader)
                
                batch_cn = next(cn_iterator)
                imgs_cn, labels_cn = batch_cn[0], batch_cn[1]
                labels_cn_4c = batch_cn[2] if len(batch_cn) > 2 else None
                
                batch_mci = next(mci_iterator)
                imgs_mci, labels_mci = batch_mci[0], batch_mci[1]
                labels_mci_4c = batch_mci[2] if len(batch_mci) > 2 else None

            imgs = torch.cat((imgs_cn, imgs_mci, imgs_ad))
            labels = torch.cat((labels_cn, labels_mci, labels_ad))
            images = imgs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            labels_4c = torch.cat((labels_cn_4c, labels_mci_4c, labels_ad_4c)).cuda(non_blocking=True) if labels_ad_4c is not None else None
            
            num_classes = model.module.num_classes if hasattr(model, 'module') else model.num_classes
            if num_classes == 4 and labels_4c is not None:
                labels = labels_4c
            
            optimizer.zero_grad()
            features, outputs, _ = model.forward(images)

            # basic computing
            compactness_loss, separation_loss, mus, triplet_ins2cls, hierarchical_triplet_ins2cls = basiccomputing(features, labels, labels_4c)

            # CE loss
            loss_CE = criterion(outputs, labels)

            # Optional QWK Loss
            if ablation_loss == 'qwk_hierarchical_triplet':
                if qwk_loss_fn is None:
                    qwk_loss_fn = DifferentiableQWKLoss(num_classes=num_classes).cuda()
                loss_QWK = qwk_loss_fn(outputs, labels)
                loss_CE = loss_QWK  # Replace standard CE with QWK


            # Hybrid-granularity ordinal loss (Ablation Control)
            loss_ins2ins = criterionRank(features, labels)  # instance-to-instance loss
            loss_ins2cls = compactness_loss / features.shape[1]  # instance-to-class loss
            
            present_classes = torch.unique(labels).float().cuda()
            if len(present_classes) > 1:
                loss_cls2cls = features.shape[1] / separation_loss + criterionRank(mus, present_classes)
            else:
                loss_cls2cls = torch.tensor(0.0, device=loss_CE.device)
            
            if ablation_loss == 'ce':
                loss_hyb = torch.tensor(0.0, device=loss_CE.device)
            elif ablation_loss == 'ins2ins':
                loss_hyb = loss_ins2ins
            elif ablation_loss == 'ins2cls':
                loss_hyb = loss_ins2ins + loss_ins2cls
            elif ablation_loss == 'exclude_ins2ins':
                loss_hyb = loss_ins2cls + loss_cls2cls
            elif ablation_loss == 'exclude_ins2cls':
                loss_hyb = loss_ins2ins + loss_cls2cls
            elif ablation_loss == 'exp_triplet_ins2cls':
                loss_hyb = loss_ins2ins + (triplet_ins2cls / features.shape[1]) + loss_cls2cls
            elif ablation_loss == 'triplet_only':
                loss_hyb = (triplet_ins2cls / features.shape[1])
            elif ablation_loss == 'exp_hierarchical_triplet_ins2cls':
                loss_hyb = loss_ins2ins + (hierarchical_triplet_ins2cls / features.shape[1]) + loss_cls2cls
            elif ablation_loss in ['hierarchical_triplet_only', 'qwk_hierarchical_triplet']:
                loss_hyb = (hierarchical_triplet_ins2cls / features.shape[1])
            else: # 'full'
                loss_hyb = loss_ins2ins + loss_ins2cls + loss_cls2cls

            # total loss
            lambda_hyb = e * (1 / epochs)
            loss = loss_CE + lambda_hyb * loss_hyb

            # backward
            loss.backward()
            optimizer.step()

            # prototype online update
            model.update(features, labels)

            # loss logging
            train_loss_CE += loss_CE.item()
            train_loss_ins2ins += loss_ins2ins.item()
            if ablation_loss in ['exp_triplet_ins2cls', 'triplet_only']:
                train_loss_ins2cls += (triplet_ins2cls / features.shape[1]).item()
            else:
                train_loss_ins2cls += loss_ins2cls.item()
            train_loss_cls2cls += loss_cls2cls.item()
            train_loss_hyb += loss_hyb.item()
            train_loss += loss.item()

            _, train_predicted = torch.max(outputs.data, 1)
            y_train_true.extend(np.ravel(np.squeeze(labels.cpu().detach().numpy())).tolist())
            y_train_pred.extend(np.ravel(np.squeeze(train_predicted.cpu().detach().numpy())).tolist())

        if scheduler:
            scheduler.step()

        val_loss = 0.
        val_samples = 0
        y_val_true = []
        y_val_pred = []
        val_prob_all = []
        
        y_val_true_4class = []
        y_val_pred_4class = []
        y_val_true_3class = []
        y_val_pred_3class = []
        
        with torch.no_grad():
            model.eval()
            for ii, (images, labels) in enumerate(valid_dataloaders):
                images, labels = images.cuda(), labels.cuda()
                _, x, outputs = model(images)
                
                _, val_predicted = torch.max(outputs.data, 1)
                _, x_predicted = torch.max(x.data, 1)
                
                for b, s, t in zip(x_predicted, val_predicted, labels):
                    t_val = t.item()
                    if num_classes == 3:
                        if b == 0:
                            pred_4c = 0
                        elif b == 2:
                            pred_4c = 3
                        else:
                            pred_4c = 1 if s == 0 else 2
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
                    mci_outputs = mci_outputs.softmax(dim=-1)
                    val_prob_all.extend(mci_outputs[:, 1].cpu().detach().numpy())

        # loss logging
        train_loss_CE = train_loss_CE / len(total_ad_loader)
        train_loss_ins2ins = train_loss_ins2ins / len(total_ad_loader)
        train_loss_ins2cls = train_loss_ins2cls / len(total_ad_loader)
        train_loss_cls2cls = train_loss_cls2cls / len(total_ad_loader)
        train_loss_hyb = train_loss_hyb / len(total_ad_loader)
        train_loss = train_loss / len(total_ad_loader)
        
        val_loss = val_loss / max(1, val_samples)
        
        val_acc_4class = accuracy_score(y_val_true_4class, y_val_pred_4class)
        val_f1_4class = f1_score(y_val_true_4class, y_val_pred_4class, average='weighted')
        val_qwk_4class = cohen_kappa_score(y_val_true_4class, y_val_pred_4class, weights='quadratic')
        
        val_acc_3class = accuracy_score(y_val_true_3class, y_val_pred_3class)
        val_f1_3class = f1_score(y_val_true_3class, y_val_pred_3class, average='weighted')
        val_qwk_3class = cohen_kappa_score(y_val_true_3class, y_val_pred_3class, weights='quadratic')
        
        # In case there are no MCI samples in the validation batch (rare but possible with very small mock datasets)
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
            
        train_acc = accuracy_score(y_train_true, y_train_pred)
        train_f1_score = f1_score(y_train_true, y_train_pred, average='weighted')
        train_recall = recall_score(y_train_true, y_train_pred, average='weighted')
        if e % print_freq == 0:
            wandb.log({"train_loss": train_loss,
                       "train_loss_hyb": train_loss_hyb,
                       "train_loss_CE": train_loss_CE,
                       "train_loss_ins2ins": train_loss_ins2ins,
                       "train_loss_ins2cls": train_loss_ins2cls,
                       "train_loss_cls2cls": train_loss_cls2cls,
                       "train_acc": train_acc,
                       "train_f1": train_f1_score,
                       "train_sen": train_recall,
                       "val_loss": val_loss,
                       "val_acc_4class": val_acc_4class,
                       "val_f1_4class": val_f1_4class,
                       "val_qwk_4class": val_qwk_4class,
                       "val_acc_3class": val_acc_3class,
                       "val_f1_3class": val_f1_3class,
                       "val_qwk_3class": val_qwk_3class,
                       "val_acc": val_acc, "val_f1": val_f1_score,
                       "val_sen": val_recall, "val_spe": val_spe,
                       "val_precision": val_precision, "val_auc": val_auc
                       })
            print('Epochs: {}/{}...'.format(e + 1, epochs),
                  'Train Loss:{:.3f}...'.format(train_loss),
                  'Train Accuracy:{:.3f}...'.format(train_acc),
                  'Val Acc 4-class:{:.3f}...'.format(val_acc_4class),
                  'Val QWK 4-class:{:.3f}...'.format(val_qwk_4class),
                  'Val Acc 3-class:{:.3f}...'.format(val_acc_3class),
                  'Val Acc 2-class (MCI):{:.3f}...'.format(val_acc)
                  )
        history.append({
            'epoch': e,
            'ablation_loss': ablation_loss,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_f1': train_f1_score,
            'val_loss': val_loss,
            'val_acc_4class': val_acc_4class,
            'val_f1_4class': val_f1_4class,
            'val_qwk_4class': val_qwk_4class,
            'val_acc_3class': val_acc_3class,
            'val_f1_3class': val_f1_3class,
            'val_qwk_3class': val_qwk_3class,
            'val_acc': val_acc,
            'val_f1': val_f1_score,
            'val_sen': val_recall,
            'val_spe': val_spe,
            'val_precision': val_precision,
            'val_auc': val_auc
        })
        
        # TRIPLE-SAVING LOGIC
        os.makedirs(expr_dir, exist_ok=True)
        if val_acc > best_val_acc_2c:
            best_val_acc_2c = val_acc
            torch.save(model.state_dict(), expr_dir + '/best_2c_net.pth')
            
        if val_acc_3class > best_val_acc_3c:
            best_val_acc_3c = val_acc_3class
            torch.save(model.state_dict(), expr_dir + '/best_3c_net.pth')
            
        if val_acc_4class > best_val_acc_4c:
            best_val_acc_4c = val_acc_4class
            torch.save(model.state_dict(), expr_dir + '/best_4c_net.pth')

    end = time.time()
    runing_time = end - start
    print('Training time is {:.0f}m {:.0f}s'.format(runing_time // 60, runing_time % 60))

    if len(history) > 0:
        # Save full history of all epochs
        history_path = os.path.join(expr_dir, 'history.csv')
        keys = history[0].keys()
        with open(history_path, mode='w', newline='') as f:
            dict_writer = csv.DictWriter(f, keys)
            dict_writer.writeheader()
            dict_writer.writerows(history)
            
        # Find best validation accuracy epoch
        best_epoch = max(history, key=lambda x: x['val_acc'])
        best_path = os.path.join(expr_dir, 'best_metrics.csv')
        with open(best_path, mode='w', newline='') as f:
            dict_writer = csv.DictWriter(f, keys)
            dict_writer.writeheader()
            dict_writer.writerow(best_epoch)
            
        print(f"Saved full history to: {history_path}")
        print(f"Saved BEST epoch (Epoch {best_epoch['epoch']}) to: {best_path}")

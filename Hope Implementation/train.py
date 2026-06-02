import os
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from models.BasicComputing import BasicComputing
from models.ranking import RankLoss
from options.train_options import TrainOptions
from utils.Dataset import *
from utils.tools import *
from utils.train_data import *

def run_fold(opt, current_fold):
    wandb.init(project="hope-replication",
               name=f"{opt.name}_fold{current_fold}",
               config={
                   "batch_size": opt.batch_size,
                   "group": opt.group,
                   "dataset": opt.dataset,
                   "learning_rate": opt.lr,
                   "architecture": opt.cls_type,
                   "epoch": opt.epoch_count,
                   "fold": current_fold
               })
    model = define_Cls(opt.cls_type, class_num=opt.class_num, init_type=opt.init_type, init_gain=opt.init_gain, m=opt.m,
                       gpu_ids=opt.gpu_ids)
    epochs = opt.epoch_count
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = get_scheduler(optimizer, opt)

    # criterion preparation
    basiccomputing = BasicComputing(class_num=opt.class_num, gpu_ids=opt.gpu_ids)
    criterion = nn.CrossEntropyLoss()
    criterionRank = RankLoss(opt.interpolation_lambda)

    # dataset preparation
    total_cn_dataset = Dataset(mode="total_cn", data_dir=opt.data_dir, seed=opt.seed, kfold=opt.kfold, current_fold=current_fold)
    total_ad_dataset = Dataset(mode="total_ad", data_dir=opt.data_dir, seed=opt.seed, kfold=opt.kfold, current_fold=current_fold)
    total_mci_dataset = Dataset(mode="total_mci", data_dir=opt.data_dir, seed=opt.seed, kfold=opt.kfold, current_fold=current_fold)
    valid_dataset = Dataset(mode="valid", data_dir=opt.data_dir, seed=opt.seed, kfold=opt.kfold, current_fold=current_fold)

    # training loader (random sample data in a stratified manner)
    # Using max(1, ...) to avoid division by zero crashes for small mock datasets
    batch_cn_ad = max(1, int(opt.batch_size / 4))
    batch_mci = max(1, int(opt.batch_size / 2))
    num_workers = max(0, int(opt.workers / 4))
    num_workers_mci = max(0, int(opt.workers / 2))

    total_cn_loader = torch.utils.data.DataLoader(
        total_cn_dataset, batch_size=batch_cn_ad, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)
    total_ad_loader = torch.utils.data.DataLoader(
        total_ad_dataset, batch_size=batch_cn_ad, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)
    total_mci_loader = torch.utils.data.DataLoader(
        total_mci_dataset, batch_size=batch_mci, shuffle=True,
        num_workers=num_workers_mci, pin_memory=True, drop_last=True)

    # valid loader
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=num_workers_mci, pin_memory=True)

    expr_dir = os.path.join(opt.checkpoints_dir, f"{opt.name}_fold{current_fold}")
    os.makedirs(expr_dir, exist_ok=True)

    # train data
    train_data(model, total_cn_loader, total_ad_loader, total_mci_loader,
               valid_loader, epochs, optimizer, scheduler,
               basiccomputing, criterion, criterionRank, expr_dir, opt.print_freq,
               opt.save_epoch_freq)
    wandb.finish()


if __name__ == '__main__':
    # -----  Loading the init options -----
    opt = TrainOptions().parse()
    
    if opt.kfold > 1:
        for f in range(1, opt.kfold + 1):
            print(f"\n{'='*40}\nStarting Fold {f}/{opt.kfold}\n{'='*40}\n")
            run_fold(opt, f)
    else:
        run_fold(opt, 1)

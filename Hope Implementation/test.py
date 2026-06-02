import torch
import torch.nn as nn

from options.test_options import TestOptions
from utils.Dataset import *
from utils.test_data import *
from utils.tools import *
from utils.train_data import *

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
    load_dir = opt.load_dir if opt.kfold == 1 else f"{opt.load_dir}_fold{current_fold}/{opt.epoch_count}_net.pth"
    try:
        state_dict = torch.load(load_dir, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        # ema prototype
        model.prototypes = state_dict['prototypes'].cuda()
        model.cuda()
        print(f"loading weights from {load_dir}")
        print("Testing on the testing set")
        test_data(model, test_loader, criterion)
    except FileNotFoundError:
        print(f"Weights {load_dir} not found. Ensure the model has been trained.")

if __name__ == '__main__':
    # -----  Loading the init options -----
    opt = TestOptions().parse()
    
    if opt.kfold > 1:
        for f in range(1, opt.kfold + 1):
            print(f"\n{'='*40}\nTesting Fold {f}/{opt.kfold}\n{'='*40}\n")
            run_test(opt, f)
    else:
        run_test(opt, 1)

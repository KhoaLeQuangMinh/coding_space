import torch
import torch.nn as nn


class BasicComputing(nn.Module):
    def __init__(self, class_num, gpu_ids=None, dim=512):
        super(BasicComputing, self).__init__()
        self.dim = dim
        self.class_num = class_num

    # compute current prototype
    def compute_mean(self, x):
        return torch.mean(x, dim=0, keepdim=True)

    # compute instance-to-class or the first term of class-to-class loss
    def compute_loss(self, x, mean):
        xg_bar = x - mean  # [None,d]
        if len(xg_bar.shape) == 1:
            xg_bar = xg_bar.unsqueeze(0)
        return torch.sum(
            torch.matmul(xg_bar.unsqueeze(dim=1), xg_bar.unsqueeze(dim=2))
            , dim=0).reshape(-1)

    # compute the new relative instance-to-class loss (triplet logic)
    def compute_relative_loss(self, x, target_mean, opp_mean, margin=0.0):
        if x.shape[0] == 0:
            return torch.tensor(0.0, device=x.device)
        dist_target = torch.sum((x - target_mean) ** 2, dim=1)
        dist_opp = torch.sum((x - opp_mean) ** 2, dim=1)
        loss = torch.nn.functional.relu(dist_target - dist_opp + margin)
        return torch.sum(loss)

    def __call__(self, features, labels, labels_4c=None):
        compactness_losslist = []
        separation_losslist = []
        all_means = []
        means_dict = {}
        
        # [0] basic definition multi-class
        for i in range(self.class_num):
            index = torch.nonzero(labels == i).reshape(-1)
            if index.numel() == 0:
                continue
            mu_k = features.index_select(0, index)
            
            mean_k = self.compute_mean(mu_k)
            all_means.append(mean_k)
            means_dict[i] = mean_k
            
            compactness_losslist.append(self.compute_loss(mu_k, mean_k))
            separation_losslist.append(self.compute_loss(mean_k, self.compute_mean(features)) * mu_k.shape[0])

        compactness_loss = sum(compactness_losslist) if len(compactness_losslist) > 0 else torch.tensor(0.0, device=features.device)
        separation_loss = sum(separation_losslist) if len(separation_losslist) > 0 else torch.tensor(0.0, device=features.device)
        
        # Ensure we return [K, D] and not [K, 1, D]
        stacked_means = torch.cat(all_means, dim=0) if len(all_means) > 0 else torch.empty((0, self.dim), device=features.device)

        # EXPERIMENTAL: Relative Distance (Triplet) Instance-to-Class Loss
        triplet_ins2cls = torch.tensor(0.0, device=features.device)
        
        if labels_4c is not None and 0 in means_dict and 2 in means_dict:
            mean_cn = means_dict[0]
            mean_ad = means_dict[2]
            
            # Left side: CN (0) and sMCI (1) -> Target CN, Opp AD
            idx_left = torch.nonzero((labels_4c == 0) | (labels_4c == 1)).reshape(-1)
            if idx_left.numel() > 0:
                x_left = features.index_select(0, idx_left)
                loss_left = self.compute_relative_loss(x_left, target_mean=mean_cn, opp_mean=mean_ad, margin=0.0)
                triplet_ins2cls += loss_left
                
            # Right side: pMCI (2) and AD (3) -> Target AD, Opp CN
            idx_right = torch.nonzero((labels_4c == 2) | (labels_4c == 3)).reshape(-1)
            if idx_right.numel() > 0:
                x_right = features.index_select(0, idx_right)
                loss_right = self.compute_relative_loss(x_right, target_mean=mean_ad, opp_mean=mean_cn, margin=0.0)
                triplet_ins2cls += loss_right

        return compactness_loss, separation_loss, stacked_means, triplet_ins2cls

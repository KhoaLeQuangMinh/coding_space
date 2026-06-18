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

    # compute the hierarchical ordinal triplet loss
    def compute_hierarchical_triplet(self, features, labels_4c):
        if labels_4c is None:
            return torch.tensor(0.0, device=features.device)
            
        means_4c = {}
        for i in range(4):
            idx = torch.nonzero(labels_4c == i).reshape(-1)
            if idx.numel() > 0:
                means_4c[i] = self.compute_mean(features.index_select(0, idx))
                
        loss = torch.tensor(0.0, device=features.device)
        
        # 1. CN should be closer to CN than sMCI
        if 0 in means_4c and 1 in means_4c:
            idx_cn = torch.nonzero(labels_4c == 0).reshape(-1)
            if idx_cn.numel() > 0:
                loss += self.compute_relative_loss(features.index_select(0, idx_cn), target_mean=means_4c[0], opp_mean=means_4c[1])
                
        # 2. sMCI should be closer to sMCI than CN
        if 1 in means_4c and 0 in means_4c:
            idx_smci = torch.nonzero(labels_4c == 1).reshape(-1)
            if idx_smci.numel() > 0:
                loss += self.compute_relative_loss(features.index_select(0, idx_smci), target_mean=means_4c[1], opp_mean=means_4c[0])

        # 3. AD should be closer to AD than pMCI
        if 3 in means_4c and 2 in means_4c:
            idx_ad = torch.nonzero(labels_4c == 3).reshape(-1)
            if idx_ad.numel() > 0:
                loss += self.compute_relative_loss(features.index_select(0, idx_ad), target_mean=means_4c[3], opp_mean=means_4c[2])

        # 4. pMCI should be closer to pMCI than AD
        if 2 in means_4c and 3 in means_4c:
            idx_pmci = torch.nonzero(labels_4c == 2).reshape(-1)
            if idx_pmci.numel() > 0:
                loss += self.compute_relative_loss(features.index_select(0, idx_pmci), target_mean=means_4c[2], opp_mean=means_4c[3])

        # 5. CN, sMCI closer to CN than AD
        if 0 in means_4c and 3 in means_4c:
            idx_left = torch.nonzero((labels_4c == 0) | (labels_4c == 1)).reshape(-1)
            if idx_left.numel() > 0:
                loss += self.compute_relative_loss(features.index_select(0, idx_left), target_mean=means_4c[0], opp_mean=means_4c[3])

        # 6. AD, pMCI closer to AD than CN
        if 3 in means_4c and 0 in means_4c:
            idx_right = torch.nonzero((labels_4c == 2) | (labels_4c == 3)).reshape(-1)
            if idx_right.numel() > 0:
                loss += self.compute_relative_loss(features.index_select(0, idx_right), target_mean=means_4c[3], opp_mean=means_4c[0])

        return loss

    def compute_3pole_triplet(self, features, labels_4c, mean_cn, mean_mci, mean_ad):
        if labels_4c is None or mean_cn is None or mean_mci is None or mean_ad is None:
            return torch.tensor(0.0, device=features.device)
            
        loss = torch.tensor(0.0, device=features.device)
        
        # 1. CN and sMCI closer to CN than AD
        idx_cn_smci = torch.nonzero((labels_4c == 0) | (labels_4c == 1)).reshape(-1)
        if idx_cn_smci.numel() > 0:
            loss += self.compute_relative_loss(features.index_select(0, idx_cn_smci), target_mean=mean_cn, opp_mean=mean_ad)
            
        # 2. AD and pMCI closer to AD than CN
        idx_ad_pmci = torch.nonzero((labels_4c == 3) | (labels_4c == 2)).reshape(-1)
        if idx_ad_pmci.numel() > 0:
            loss += self.compute_relative_loss(features.index_select(0, idx_ad_pmci), target_mean=mean_ad, opp_mean=mean_cn)
            
        # 3. sMCI closer to MCI than CN
        idx_smci = torch.nonzero(labels_4c == 1).reshape(-1)
        if idx_smci.numel() > 0:
            loss += self.compute_relative_loss(features.index_select(0, idx_smci), target_mean=mean_mci, opp_mean=mean_cn)
            
        # 4. pMCI closer to MCI than AD
        idx_pmci = torch.nonzero(labels_4c == 2).reshape(-1)
        if idx_pmci.numel() > 0:
            loss += self.compute_relative_loss(features.index_select(0, idx_pmci), target_mean=mean_mci, opp_mean=mean_ad)
            
        return loss

    def __call__(self, features, labels, labels_4c=None, global_protos=None):
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
        
        if labels_4c is not None and 0 in means_dict and (self.class_num - 1) in means_dict:
            mean_cn = means_dict[0]
            mean_ad = means_dict[self.class_num - 1]
            
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

        # EXPERIMENTAL 2: Hierarchical Ordinal Triplet Loss
        hierarchical_triplet_ins2cls = self.compute_hierarchical_triplet(features, labels_4c)

        # EXPERIMENTAL 3: 3-Pole Ordinal Triplet Loss (Local)
        three_pole_local = torch.tensor(0.0, device=features.device)
        if self.class_num == 3 and labels_4c is not None and 0 in means_dict and 1 in means_dict and 2 in means_dict:
            three_pole_local = self.compute_3pole_triplet(features, labels_4c, means_dict[0], means_dict[1], means_dict[2])

        # EXPERIMENTAL 4: 3-Pole Ordinal Triplet Loss (Global)
        three_pole_global = torch.tensor(0.0, device=features.device)
        if self.class_num == 3 and labels_4c is not None and global_protos is not None and global_protos.shape[0] >= 3:
            three_pole_global = self.compute_3pole_triplet(features, labels_4c, global_protos[0], global_protos[1], global_protos[2])

        return compactness_loss, separation_loss, stacked_means, triplet_ins2cls, hierarchical_triplet_ins2cls, three_pole_local, three_pole_global

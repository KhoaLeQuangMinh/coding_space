import torch
import torch.nn as nn


class BasicComputing(nn.Module):
    def __init__(self, class_num, gpu_ids=None, dim=512, margin=0.0, m=0.9, intra_margin=0.15):
        super(BasicComputing, self).__init__()
        self.dim = dim
        self.class_num = class_num
        self.margin = margin
        self.m = m
        self.intra_margin = intra_margin
        
        # Register buffers for running statistics of distances to prototypes (EMA stats)
        # Note: class_num represents the number of classes (typically 3)
        self.register_buffer("running_mean_dist", torch.zeros(class_num))
        self.register_buffer("running_mean_sq_dist", torch.zeros(class_num))

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
    def compute_relative_loss(self, x, target_mean, opp_mean, margin=None):
        if margin is None:
            margin = self.margin
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

    def compute_intra_pole_loss(self, features, labels_4c, global_protos, margin=None):
        if labels_4c is None or global_protos is None or global_protos.shape[0] < self.class_num:
            return torch.tensor(0.0, device=features.device)
        if margin is None:
            margin = self.intra_margin
            
        loss = torch.tensor(0.0, device=features.device)
        
        idx_cn = torch.nonzero(labels_4c == 0).reshape(-1)
        idx_smci = torch.nonzero(labels_4c == 1).reshape(-1)
        idx_pmci = torch.nonzero(labels_4c == 2).reshape(-1)
        idx_ad = torch.nonzero(labels_4c == 3).reshape(-1)
        
        # 1. sMCI further from CN prototype than CN
        if idx_cn.numel() > 0 and idx_smci.numel() > 0:
            dist_cn = torch.sum((features.index_select(0, idx_cn) - global_protos[0]) ** 2, dim=1)
            dist_smci = torch.sum((features.index_select(0, idx_smci) - global_protos[0]) ** 2, dim=1)
            loss_cn_smci = torch.nn.functional.relu(dist_cn.unsqueeze(1) - dist_smci.unsqueeze(0) + margin)
            loss += loss_cn_smci.mean(dim=0).sum()
            
        # 2. pMCI further from AD prototype than AD
        if idx_ad.numel() > 0 and idx_pmci.numel() > 0:
            proto_ad = global_protos[self.class_num - 1]
            dist_ad = torch.sum((features.index_select(0, idx_ad) - proto_ad) ** 2, dim=1)
            dist_pmci = torch.sum((features.index_select(0, idx_pmci) - proto_ad) ** 2, dim=1)
            loss_ad_pmci = torch.nn.functional.relu(dist_ad.unsqueeze(1) - dist_pmci.unsqueeze(0) + margin)
            loss += loss_ad_pmci.mean(dim=0).sum()
            
        return loss

    def compute_intra_pole_loss_distributional(self, features, labels_4c, global_protos, k=None):
        if labels_4c is None or global_protos is None or global_protos.shape[0] < self.class_num:
            return torch.tensor(0.0, device=features.device)
        if k is None:
            k = self.intra_margin
            
        loss = torch.tensor(0.0, device=features.device)
        
        idx_cn = torch.nonzero(labels_4c == 0).reshape(-1)
        idx_smci = torch.nonzero(labels_4c == 1).reshape(-1)
        idx_pmci = torch.nonzero(labels_4c == 2).reshape(-1)
        idx_ad = torch.nonzero(labels_4c == 3).reshape(-1)
        
        # 1. Update running stats for CN and calculate sMCI separation loss
        if idx_cn.numel() > 0:
            dist_cn = torch.sum((features.index_select(0, idx_cn).detach() - global_protos[0]) ** 2, dim=1)
            m1_cn = dist_cn.mean()
            m2_cn = (dist_cn ** 2).mean()
            if self.running_mean_dist[0].item() == 0.0:
                self.running_mean_dist[0] = m1_cn
                self.running_mean_sq_dist[0] = m2_cn
            else:
                self.running_mean_dist[0] = self.running_mean_dist[0] * self.m + (1.0 - self.m) * m1_cn
                self.running_mean_sq_dist[0] = self.running_mean_sq_dist[0] * self.m + (1.0 - self.m) * m2_cn
                
        s1_cn = self.running_mean_dist[0]
        s2_cn = self.running_mean_sq_dist[0]
        var_cn = torch.clamp(s2_cn - s1_cn ** 2, min=0.0)
        std_cn = torch.sqrt(var_cn + 1e-6)
        threshold_cn = s1_cn + k * std_cn
        
        if idx_smci.numel() > 0:
            dist_smci = torch.sum((features.index_select(0, idx_smci) - global_protos[0]) ** 2, dim=1)
            loss_smci = torch.nn.functional.relu(threshold_cn - dist_smci)
            loss += loss_smci.mean()
            
        # 2. Update running stats for AD and calculate pMCI separation loss
        proto_ad = global_protos[self.class_num - 1]
        if idx_ad.numel() > 0:
            dist_ad = torch.sum((features.index_select(0, idx_ad).detach() - proto_ad) ** 2, dim=1)
            m1_ad = dist_ad.mean()
            m2_ad = (dist_ad ** 2).mean()
            if self.running_mean_dist[self.class_num - 1].item() == 0.0:
                self.running_mean_dist[self.class_num - 1] = m1_ad
                self.running_mean_sq_dist[self.class_num - 1] = m2_ad
            else:
                self.running_mean_dist[self.class_num - 1] = self.running_mean_dist[self.class_num - 1] * self.m + (1.0 - self.m) * m1_ad
                self.running_mean_sq_dist[self.class_num - 1] = self.running_mean_sq_dist[self.class_num - 1] * self.m + (1.0 - self.m) * m2_ad
                
        s1_ad = self.running_mean_dist[self.class_num - 1]
        s2_ad = self.running_mean_sq_dist[self.class_num - 1]
        var_ad = torch.clamp(s2_ad - s1_ad ** 2, min=0.0)
        std_ad = torch.sqrt(var_ad + 1e-6)
        threshold_ad = s1_ad + k * std_ad
        
        if idx_pmci.numel() > 0:
            dist_pmci = torch.sum((features.index_select(0, idx_pmci) - proto_ad) ** 2, dim=1)
            loss_pmci = torch.nn.functional.relu(threshold_ad - dist_pmci)
            loss += loss_pmci.mean()
            
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
                loss_left = self.compute_relative_loss(x_left, target_mean=mean_cn, opp_mean=mean_ad)
                triplet_ins2cls += loss_left
                
            # Right side: pMCI (2) and AD (3) -> Target AD, Opp CN
            idx_right = torch.nonzero((labels_4c == 2) | (labels_4c == 3)).reshape(-1)
            if idx_right.numel() > 0:
                x_right = features.index_select(0, idx_right)
                loss_right = self.compute_relative_loss(x_right, target_mean=mean_ad, opp_mean=mean_cn)
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

        # EXPERIMENTAL 5: Collinearity Loss on candidate prototypes
        collinear_loss = torch.tensor(0.0, device=features.device)
        P_prime = torch.zeros_like(global_protos) if global_protos is not None else None
        
        if global_protos is not None and global_protos.shape[0] >= 3:
            means_dict_full = {}
            for c in range(self.class_num):
                index = torch.nonzero(labels == c).reshape(-1)
                if index.numel() > 0:
                    means_dict_full[c] = torch.nn.functional.normalize(features.index_select(0, index).mean(dim=0), p=2, dim=0)
                else:
                    means_dict_full[c] = global_protos[c]
            
            for c in range(self.class_num):
                P_prime[c] = torch.nn.functional.normalize(
                    global_protos[c] * self.m + (1.0 - self.m) * means_dict_full[c], p=2, dim=0
                )
                
            midpoint = (P_prime[0] + P_prime[2]) / 2.0
            collinear_loss = torch.sum((P_prime[1] - midpoint) ** 2)

        # Compute Intra-Pole Triplet Loss
        intra_pole_loss = self.compute_intra_pole_loss(features, labels_4c, global_protos)
        
        # Compute Distributional Running-Stats Asymmetric Intra-Pole Loss
        intra_pole_loss_dist = self.compute_intra_pole_loss_distributional(features, labels_4c, global_protos)

        # EXPERIMENTAL 6: Global Prototype Anchored Triplet Loss
        triplet_ins2cls_global = torch.tensor(0.0, device=features.device)
        if labels_4c is not None and global_protos is not None and global_protos.shape[0] >= self.class_num:
            mean_cn = global_protos[0].unsqueeze(0)
            mean_ad = global_protos[self.class_num - 1].unsqueeze(0)
            
            # Left side: CN (0) and sMCI (1) -> Target CN, Opp AD
            idx_left = torch.nonzero((labels_4c == 0) | (labels_4c == 1)).reshape(-1)
            if idx_left.numel() > 0:
                x_left = features.index_select(0, idx_left)
                loss_left = self.compute_relative_loss(x_left, target_mean=mean_cn, opp_mean=mean_ad)
                triplet_ins2cls_global += loss_left
                
            # Right side: pMCI (2) and AD (3) -> Target AD, Opp CN
            idx_right = torch.nonzero((labels_4c == 2) | (labels_4c == 3)).reshape(-1)
            if idx_right.numel() > 0:
                x_right = features.index_select(0, idx_right)
                loss_right = self.compute_relative_loss(x_right, target_mean=mean_ad, opp_mean=mean_cn)
                triplet_ins2cls_global += loss_right

        return compactness_loss, separation_loss, stacked_means, triplet_ins2cls, hierarchical_triplet_ins2cls, three_pole_local, three_pole_global, collinear_loss, P_prime, intra_pole_loss, intra_pole_loss_dist, triplet_ins2cls_global

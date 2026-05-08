import torch
import torch.nn as nn
from vit3d import vit_b16_backbone 
from src.fusion_modules import ConcatFusion, SumFusion, FiLM, GatedFusion, CrossAttention, ClinicalGuideCrossAttention

def create_vit_backbone(pretrained=False, pretrained_path=None):
    model = vit_b16_backbone()
    if pretrained and pretrained_path:
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(checkpoint['net'], strict=False)
    return model

class MriClassifier(nn.Module):
    """
    Mri modality classifier
    """

    def __init__(self, mri_model, out_feature_dim, class_num):
        super(MriClassifier, self).__init__()
        self.net = mri_model
        self.classifier = nn.Linear(in_features=out_feature_dim, out_features=class_num)

    def forward(self, x):
        m = self.net(x)
        output = self.classifier(m)
        return m, output


class PetClassifier(nn.Module):
    """
    Pet modality classifier
    """

    def __init__(self, pet_model, out_feature_dim, class_num):
        super(PetClassifier, self).__init__()
        self.net = pet_model
        self.classifier = nn.Linear(in_features=out_feature_dim, out_features=class_num)

    def forward(self, x):
        p = self.net(x)
        output = self.classifier(p)
        return p, output

class BaselineModel(nn.Module):
    """
    traditional joint multimodal learning
    """

    def __init__(self, fusion_method = "concat", out_feature_dim = 768, class_num = 4, pretrained=False, pretrained_path=None):
        super(BaselineModel, self).__init__()
        mri_backbone = create_vit_backbone(pretrained=pretrained, pretrained_path=pretrained_path)
        pet_backbone = create_vit_backbone(pretrained=pretrained, pretrained_path=pretrained_path)
        
        self.mri_model = MriClassifier(mri_backbone, out_feature_dim=out_feature_dim, class_num=class_num)
        self.pet_model = PetClassifier(pet_backbone, out_feature_dim=out_feature_dim, class_num=class_num)

        if fusion_method == 'sum':
            self.fusion_module = SumFusion(input_dim=out_feature_dim, output_dim=class_num)
        elif fusion_method == 'concat':
            self.fusion_module = ConcatFusion(input_dim=out_feature_dim, output_dim=class_num)
        elif fusion_method == 'film':
            self.fusion_module = FiLM(input_dim=out_feature_dim, output_dim=class_num, x_film=True)
        elif fusion_method == 'gated':
            self.fusion_module = GatedFusion(input_dim=out_feature_dim, output_dim=class_num, x_gate=True)
        elif fusion_method == 'CrossAttention':
            self.fusion_module = CrossAttention(input_dim=out_feature_dim, output_dim=class_num)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion_method))

    def forward(self, mri, pet):
        mri_feature, _ = self.mri_model(mri)
        pet_feature, _ = self.pet_model(pet)

        _, _, output = self.fusion_module(mri_feature, pet_feature)

        return output
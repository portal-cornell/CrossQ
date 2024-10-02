import torch
from torchvision import transforms
from torch import nn

from vlm_reward.reward_models.resnet import resnet50_backbone as resnet50

class BackboneHeadReco(torch.nn.Module):
    def __init__(self, backbone, head, reconstruct):
        super(BackboneHeadReco, self).__init__()
        self.backbone = backbone
        self.head = head
        self.reconstruct = reconstruct
        
    def forward(self, x):
        emb = self.backbone.forward_emb(x)
        pred = self.head(emb)
        emb_reco = self.reconstruct(emb)
        return pred, emb, emb_reco

def load_joint_prediction_model(device, checkpoint, output_dim, is_reco=False):
    if is_reco:
        return load_resnet50_reco(device, checkpoint, output_dim)
    else:
        return load_resnet50(device, checkpoint, output_dim)

def load_resnet50(device, checkpoint, output_dim):
    """
    Load a resnet50 pretrained model. model_cfg should contain output_dim, checkpoint_path

    Inputs:
        pretrained: if the model should pretrained (on ImageNetV2) or not
    Outputs:
        torch.nn.Module: resnet50 torch model
        torch.transforms.Transform: transform to apply to PIL images before input
    """
    # Initialize pre-trained model
    model = resnet50(pretrained=False)

    # Replace the class prediction head with an embedding prediction head
    model.fc = nn.Linear(model.fc.in_features, output_dim) 

    state_dict = torch.load(checkpoint)
    model.load_state_dict(state_dict)

    # from the resnet50 code, this is the image format expected
    # transform does not convert to tensor (since we are not loading images from paths here)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]) 

    model.to(device) 
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()

    return model, transform

def load_resnet50_reco(device, checkpoint, output_dim):
    """
    Load a resnet50 pretrained model

    Inputs:
        pretrained: if the model should pretrained (on ImageNetV2) or not
    Outputs:
        torch.nn.Module: resnet50 torch model
        torch.transforms.Transform: transform to apply to PIL images before input
    """
    # model = resnet50(pretrained=pretrained)
    # model.fc = nn.Linear(model.fc.in_features, output_dim)
    # Initialize pre-trained model

    backbone = resnet50(pretrained=False)
    embed_dim = backbone.fc.in_features
    hidden_dim = backbone.fc.in_features // 2
    reco_bottleneck_dim = 32
    
    # 3 layer MLP head
    head= nn.Sequential(
        nn.Linear(embed_dim, hidden_dim),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Linear(hidden_dim, output_dim),
    ) 

    # 2 layer bottleneck autoencoder
    reconstruct = nn.Sequential(
        nn.Linear(embed_dim, reco_bottleneck_dim),
        nn.Linear(reco_bottleneck_dim, embed_dim),
    ) 
    
    model = BackboneHeadReco(backbone, head, reconstruct)

    if checkpoint is not None:
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict, strict=False)
        print("loaded state dict")
        
    # from the resnet50 code, this is the image format expected
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]) 

    model.to(device) 
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()

    return model, transform

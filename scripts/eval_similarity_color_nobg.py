"""
Evaluate models on the preference set of Mujoco (no background + coloring)
"""
import os
from dreamsim import dreamsim
from PIL import Image
import torch
import torch.nn.functional as F

from torchvision import transforms
import pandas as pd
import functools
import numpy as np
import json

 ROOT = "../data/similarity_color_nobg"

data = os.listdir(ROOT)


# For dreamsim
def compute_dreamsim_score(model, preprocess, anchor_path, image_path, device):
    anchor = preprocess(Image.open(anchor_path)).to(device)
    image = preprocess(Image.open(image_path)).to(device)
    return model(anchor, image).item()

def compute_ds_similarity_score(model, preprocess, anchor_path, image_path, device):
    """Because DreamSim is a distance metric"""
    ds_score = compute_dreamsim_score(model, preprocess, anchor_path, image_path, device)
    return 1 - ds_score


# For Dinov2
def extract_features(model, preprocess, image_path, device):
    img = Image.open(image_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        features = model(img_tensor)
    
    return features

def compute_cosine_similarity(features1, features2):
    return F.cosine_similarity(features1, features2, dim=1).squeeze()

def compute_dinov2_similarity(model, preprocess, anchor_path, image_path, device):
    anchor_features = extract_features(model, preprocess, anchor_path, device)
    image_features = extract_features(model, preprocess, image_path, device)
    
    similarity = compute_cosine_similarity(anchor_features, image_features)
    return similarity.item()


def main(similarity_method='dreamsim'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if similarity_method == 'dreamsim':
        model, preprocess = dreamsim(pretrained=True)
        model = model.to(device)
        compute_similarity = compute_ds_similarity_score
    elif similarity_method == 'dinov2_vitl14_reg':
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
        model = model.to(device)
        model.eval()
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        compute_similarity = compute_dinov2_similarity
    else:
        raise ValueError("Invalid similarity method")

    anchors = ["left_up.png", "left_up_color.png"]
    candidates_no_color = ["left_up_2.png", "left_up_train.png", "right_up.png", "both_up.png"]
    candidates_color = ["left_up_2_color.png", "left_up_train_color.png", "right_up_color.png", "both_up_color.png"]

    results = {}

    for anchor in anchors:
        anchor_path = os.path.join(ROOT, anchor)
        results[anchor] = {"no_color": {}, "color": {}}

        for candidate in candidates_no_color:
            candidate_path = os.path.join(ROOT, candidate)
            score = compute_similarity(model, preprocess, anchor_path, candidate_path, device)
            results[anchor]["no_color"][candidate] = score

        for candidate in candidates_color:
            candidate_path = os.path.join(ROOT, candidate)
            score = compute_similarity(model, preprocess, anchor_path, candidate_path, device)
            results[anchor]["color"][candidate] = score

    for anchor, scores in results.items():
        print(f"\nAnalysis for anchor: {anchor}")
        
        print("No color scores:")
        for candidate in candidates_no_color:
            score = scores['no_color'][candidate]
            print(f"  {candidate}: {score:.4f}")
        print(f"Average score with no color: {np.mean(list(scores['no_color'].values())):.4f}")
        no_color_order = sorted(candidates_no_color, key=lambda x: scores['no_color'][x], reverse=True)
        no_color_order = ">".join([str(candidates_no_color.index(c) + 1) for c in no_color_order])
        print(f"Ordering: {no_color_order}")
        
        print("\nColor scores:")
        for candidate in candidates_color:
            score = scores['color'][candidate]
            print(f"  {candidate}: {score:.4f}")
        print(f"Average score with color: {np.mean(list(scores['color'].values())):.4f}")
        color_order = sorted(candidates_color, key=lambda x: scores['color'][x], reverse=True)
        color_order = ">".join([str(candidates_color.index(c) + 1) for c in color_order])
        print(f"Ordering: {color_order}")
        
        diff = np.mean(list(scores['color'].values())) - np.mean(list(scores['no_color'].values()))
        print(f"\nDifference (color - no color): {diff:.4f}")

    with open(f"results_{similarity_method}.json", "w") as fout:
        json.dump(results, fout, indent=4)



if __name__ == "__main__":
    # MODEL = 'dreamsim'
    MODEL = 'dinov2_vitl14_reg'
    main(MODEL)

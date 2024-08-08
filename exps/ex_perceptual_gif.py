import torch
import os
import imageio
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import lpips
from dreamsim import dreamsim

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load DreamSim model
model_dreamsim, preprocess_dreamsim = dreamsim(pretrained=True)

# Load LPIPS models
loss_fn_alex = lpips.LPIPS(net='alex').to(device)  # AlexNet variant

# Image transformation for LPIPS (normalize to [-1, 1])
transform_lpips = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2.0 - 1.0)
])

# Function to extract frames from a GIF
def extract_frames(gif_path):
    gif = imageio.mimread(gif_path)
    frames = [Image.fromarray(frame) for frame in gif]
    frames = [frame.convert('RGB') for frame in frames]
    return frames

# Compute DreamSim similarity
def compute_dreamsim_similarity(frames, target_image_path, model, preprocess):
    target_image = preprocess(Image.open(target_image_path).convert('RGB')).to(device)
    similarity_scores = []
    for frame in frames:
        frame_tensor = preprocess(frame).to(device)
        distance = model(target_image, frame_tensor)
        similarity = 1.0 / (1.0 + distance)  # Inverse distance as similarity
        similarity_scores.append(similarity.item())
    return similarity_scores

# Compute LPIPS similarity
def compute_lpips_similarity(frames, target_image_path, model, transform):
    target_image = transform(Image.open(target_image_path).convert('RGB')).unsqueeze(0).to(device)
    similarity_scores = []
    for frame in frames:
        frame_tensor = transform(frame).unsqueeze(0).to(device)
        distance = model(target_image, frame_tensor)
        similarity = 1.0 / (1.0 + distance)  # Inverse distance as similarity
        similarity_scores.append(similarity.item())
    return similarity_scores

# Plot the evolution of similarity scores for all GIFs
def plot_similarity(all_scores, metric_names, gif_names, output_path):
    plt.figure(figsize=(12, 6))
    for scores, metric_name in zip(all_scores, metric_names):
        for score, gif_name in zip(scores, gif_names):
            plt.plot(score, label=f'{gif_name} - {metric_name}')
    plt.xlabel('Frame')
    plt.ylabel('Similarity')
    plt.title('Similarity between target image and GIF frames using DreamSim')
    plt.legend()
    plt.savefig(output_path)
    plt.close()

# Paths to the GIFs and the target image
gif_root = "/home/aw588/git_annshin/CrossQ_yuki/axis_exp/kneeling_gifs"
gif_paths = [f"{gif_root}/0_success_crossq_kneel.gif",
                f"{gif_root}/1_kneel-at-20_fall-backward.gif",
                f"{gif_root}/2_some-move-close-to-kneeling.gif",]
#                 f"{gif_root}/3_crossq_stand_never-on-ground.gif",]
# RuntimeError: imageio.mimread() has read over 256000000B of image data.
# Stopped to avoid memory problems. Use imageio.get_reader(), increase threshold, or memtest=False

target_image_path = "/home/aw588/git_annshin/CrossQ/preference_data/kneeling_success_frame45.png"
output_path = "test_perceptual_similarity_plot_dreamsim.png"

# Process each GIF and plot the similarities
all_scores = []
metric_names = ['DreamSim'] #['LPIPS'] # 'DreamSim', 
for metric_name in metric_names:
    metric_scores = []
    for gif_path in gif_paths:
        print(f"Processing {gif_path} with {metric_name}")
        frames = extract_frames(gif_path)
        if metric_name == 'DreamSim':
            similarity_scores = compute_dreamsim_similarity(frames, target_image_path, model_dreamsim, preprocess_dreamsim)
        elif metric_name == 'LPIPS':
            similarity_scores = compute_lpips_similarity(frames, target_image_path, loss_fn_alex, transform_lpips)
        metric_scores.append(similarity_scores)
    all_scores.append(metric_scores)

# Save the plot to disk
plot_similarity(all_scores, metric_names, [os.path.basename(p) for p in gif_paths], output_path)
print(f"Plot saved to {output_path}")

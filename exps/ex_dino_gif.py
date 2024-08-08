import torch
import os
import imageio
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg').to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_frames(gif_path):
    gif = imageio.mimread(gif_path)
    frames = [Image.fromarray(frame) for frame in gif]
    frames = [frame.convert('RGB') for frame in frames]
    return frames

def compute_embeddings(frames, target_image_path, model, transform):
    target_image = Image.open(target_image_path).convert('RGB')
    target_image = transform(target_image).unsqueeze(0).to(device)
    with torch.no_grad():
        target_embedding = model(target_image)
    target_embedding /= target_embedding.norm(dim=-1, keepdim=True)
    
    frame_embeddings = []
    for frame in frames:
        frame_tensor = transform(frame).unsqueeze(0).to(device)
        with torch.no_grad():
            frame_embedding = model(frame_tensor)
        frame_embedding /= frame_embedding.norm(dim=-1, keepdim=True)
        frame_embeddings.append(frame_embedding.cpu().numpy())
    
    return frame_embeddings, target_embedding.cpu().numpy()

def compute_cosine_similarity(frame_embeddings, target_embedding):
    similarity_scores = []
    for frame_embedding in frame_embeddings:
        similarity = np.dot(target_embedding, frame_embedding.T)
        similarity_scores.append(similarity[0][0])
    return similarity_scores

# # Single gif
# def plot_similarity(scores, output_path):
#     plt.figure(figsize=(12, 6))
#     plt.plot(scores, label='Similarity')
#     plt.xlabel('Frame')
#     plt.ylabel('Cosine similarity')
#     plt.title('Cosine similarity between target image and GIF frames')
#     plt.legend()
#     plt.savefig(output_path)
#     plt.close()

def plot_similarity(all_scores, gif_names, output_path):
    plt.figure(figsize=(12, 6))
    for scores, gif_name in zip(all_scores, gif_names):
        plt.plot(scores, label=gif_name)
    plt.xlabel('Frame')
    plt.ylabel('Cosine similarity')
    plt.title('Cosine similarity between target image and GIF frames')
    plt.legend()
    plt.savefig(output_path)
    plt.close()



# Paths to the GIFs and the target image
gif_root = "/home/aw588/git_annshin/CrossQ_yuki/axis_exp/kneeling_gifs"
gif_paths = [f"{gif_root}/0_success_crossq_kneel.gif",
                f"{gif_root}/1_kneel-at-20_fall-backward.gif",
                f"{gif_root}/2_some-move-close-to-kneeling.gif",]

gif_path = gif_paths[0]
target_image_path = "/home/aw588/git_annshin/CrossQ/preference_data/kneeling_success_frame45.png"
output_path = "test_similarity_plot.png"

# Process each GIF and plot the similarities
all_scores = []
gif_names = []
for gif_path in gif_paths:
    print(f"Processing {gif_path}")
    frames = extract_frames(gif_path)
    frame_embeddings, target_embedding = compute_embeddings(frames, target_image_path, model, transform)
    similarity_scores = compute_cosine_similarity(frame_embeddings, target_embedding)
    all_scores.append(similarity_scores)
    gif_names.append(os.path.basename(gif_path))

# Save the plot to disk
plot_similarity(all_scores, gif_names, output_path)
print(f"Plot saved to {output_path}")
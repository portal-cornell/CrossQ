import torch
import os
import requests
from PIL import Image
from transformers import SiglipProcessor, SiglipModel
from transformers import AutoProcessor, AutoModel
import imageio
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model and processor
model = AutoModel.from_pretrained(
    # "google/siglip-so400m-patch14-384",
    "google/siglip-base-patch16-224",
    # "openai/clip-vit-large-patch14",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map=device,
)
# processor = SiglipProcessor.from_pretrained("google/siglip-so400m-patch14-384")
# processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
processor = SiglipProcessor.from_pretrained("google/siglip-base-patch16-224")

# Function to extract frames from a GIF
def extract_frames(gif_path):
    gif = imageio.mimread(gif_path)
    frames = [Image.fromarray(frame) for frame in gif]
    # Ensure all frames are in 'RGB' mode
    frames = [frame.convert('RGB') for frame in frames]
    return frames

# Function to compute embeddings for frames and text
def compute_embeddings(frames, sentence, model, processor):
    frame_embeddings = []
    for frame in frames:
        inputs = processor(images=frame, return_tensors="pt").to(device)
        with torch.no_grad():
            with torch.autocast(device):
                import pdb; pdb.set_trace()
                frame_embedding = model.get_image_features(**inputs).float()
        frame_embedding /= frame_embedding.norm(dim=-1, keepdim=True)
        frame_embeddings.append(frame_embedding.cpu().numpy())
    
    text_inputs = processor(text=[sentence], return_tensors="pt").to(device)
    with torch.no_grad():
        with torch.autocast(device):
            text_embedding = model.get_text_features(**text_inputs).float()
    text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
    
    return frame_embeddings, text_embedding.cpu().numpy()

# Function to compute cosine similarity
def compute_cosine_similarity(frame_embeddings, text_embedding):
    similarity_scores = []
    for frame_embedding in frame_embeddings:
        similarity = np.dot(text_embedding, frame_embedding.T)
        similarity_scores.append(similarity[0][0])
    return similarity_scores

# Plot the evolution of similarity scores
def plot_similarity(scores_list, gif_names, output_path):
    plt.figure(figsize=(12, 6))
    for scores, gif_name in zip(scores_list, gif_names):
        plt.plot(scores, label=gif_name)
    plt.xlabel('Frame')
    plt.ylabel('Cosine similarity')
    plt.title('Cosine similarity between sentence and GIF frames')
    plt.legend()
    plt.savefig(output_path)
    plt.close()

# Paths to the GIFs and the sentence
gif_root = "/home/aw588/git_annshin/CrossQ_yuki/axis_exp/kneeling_gifs"
gif_paths = [f"{gif_root}/0_success_crossq_kneel.gif",
                f"{gif_root}/1_kneel-at-20_fall-backward.gif",
                f"{gif_root}/2_some-move-close-to-kneeling.gif",]
# gif_paths = [f"{gif_root}/3_crossq_stand_never-on-ground.gif"]
sentence = "a humanoid robot kneeling"

# Process each GIF
similarity_scores_list = []
gif_names = []
for gif_path in gif_paths:
    print(f"Processing {gif_path}")
    gif_name = gif_path.split("/")[-1].split(".")[0]
    frames = extract_frames(gif_path)
    frame_embeddings, text_embedding = compute_embeddings(frames, sentence, model, processor)
    similarity_scores = compute_cosine_similarity(frame_embeddings, text_embedding)
    similarity_scores_list.append(similarity_scores)
    gif_names.append(os.path.basename(gif_path))  # Extract the file name for labeling

    output_path = f"{gif_name}_similarity.png"
    # plot_similarity(similarity_scores_list, gif_names, output_path)


# # Save the plot to disk
# output_path = "similarity_plot.png"
# plot_similarity(similarity_scores_list, gif_names, output_path)

# print(f"Plot saved to {output_path}")

# output = model.vision_model(**inputs)
# -- so400m, 14, 384
# output.last_hidden_state.shape
# torch.Size([1, 729, 1152]) 
# (Pdb) output.pooler_output.shape: torch.Size([1, 1152])
# -- base, 16, 224
# output.last_hidden_state.shape

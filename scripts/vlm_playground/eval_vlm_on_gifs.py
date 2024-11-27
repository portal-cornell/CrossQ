import os
import argparse
from tqdm import tqdm
import imageio
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torchvision.transforms.functional import pil_to_tensor
import numpy as np

# VLM imports
from vlm_reward.utils.dino_reward_model import Dino2FeatureExtractor
from seq_reward.cost_fns import cosine_distance
from seq_reward.seq_utils import plot_matrix_as_heatmap_on_ax

DIST_FN_DICT = {
    "cosine": cosine_distance
}

def load_frames_to_torch(gif_path, skip_frames=10):
    """
    Parameters:
        gif_path (str): the path to the gif file
        skip_frames (int): the number of frames to skip
    
    Returns:
        torch_frames (Torch.Tensor): the frames of the gif file as a Torch.Tensor
            shape: (num_frames, C, H, W)
        frames (List[PIL.Image]): the frames of the gif file as a list of PIL.Image
    """
    gif_obj = Image.open(gif_path)
    frames = [gif_obj.seek(frame_index) or gif_obj.convert("RGB") for frame_index in range(gif_obj.n_frames)]

    # Skip frames (But include the second to last frame)
    #   This is to avoid the last frame which is usually a reset of the environment
    frames = frames[::skip_frames] + [frames[-2]]

    torch_frames = torch.stack([pil_to_tensor(frame) for frame in frames])

    return torch_frames, frames

def load_vlm_model_fn(vlm_model_name):
    """
    Parameters:
        vlm_model_name (str): the VLM model name required to load it

    Returns:
        preprocess_image (Callable): the function to preprocess images
        forward_pass (Callable): the function to pass the images through the VLM model
    """
    if "dino" in vlm_model_name.lower():
        dino_model = Dino2FeatureExtractor(
            model_name=vlm_model_name,
            edge_size=224,  # The default value
        )

        def preprocess_image(images):
            """
            Parameters:
                images (PIL.Images or Torch.Tensor): the input image to be preprocessed
            """
            return dino_model.prepare_images_parallel(images)
        
        def forward_pass(batch_images):
            """
            Parameters:
                batch_images (Torch.Tensor): a batch of images to be processed
            """
            return dino_model.extract_features_final(batch_images)
        
        return preprocess_image, forward_pass
            

def run_vlm_on_gif_path(frames_tensor, preprocess_fn, forward_pass_fn, batch_size=16):
    """ Compute the VLM features for each frame in the gif file

    Parameters:
        frames_tensor (Torch.Tensor): the frames of the gif file
            shape: (num_frames, C, H, W)
        vlm_model (nn.Module): the VLM model to be used

    Returns:
        all_features (Torch.Tensor): the VLM features for each frame in the gif
            shape: (num_frames, feature_dim)
    """
    preprocessed_frames = preprocess_fn(frames_tensor)

    all_features = []

    # Split the frames into batches and pass them through the VLM model
    for batch_frames in torch.split(preprocessed_frames, batch_size):
        features = forward_pass_fn(batch_frames)

        all_features.append(features)

    all_features = torch.cat(all_features, dim=0)

    return all_features


def compute_frame_to_frame_distance(seq_1_features, seq_2_features, distance_fn):
    """
    Parameters:
        seq_1_features (Torch.Tensor): the VLM features for the first sequence
            shape: (num_frames_1, feature_dim)
        seq_2_features (Torch.Tensor): the VLM features for the second sequence
            shape: (num_frames_2, feature_dim)
        distance_fn (Callable): the distance function to be used to compute the distance between two frames

    Returns:
        frame_to_frame_distance (Torch.Tensor): the distance between each pair of frames
            shape: (num_frames_1, num_frames_2)
    """
    # Detach the tensors and convert them to numpy arrays
    seq_1_features = seq_1_features.detach().cpu().numpy()
    seq_2_features = seq_2_features.detach().cpu().numpy()
    return distance_fn(seq_1_features, seq_2_features)


def plot_distance_matrix(distance_matrix, frames, save_fig_path, fig_name, rolcol_size=1):
    """
    Parameters:
        distance_matrix (Torch.Tensor): the distance between each pair of frames
            shape: (num_frames, num_frames)
        frames (List[PIL.Image]): the frames of the gif file as a list of PIL.Image
    """
    fig_width = rolcol_size * (len(frames) + 1)
    fig_height = rolcol_size * (len(frames) + 1)

    fig, axs = plt.subplots(1, 1, figsize=(fig_width, fig_height))

    plot_matrix_as_heatmap_on_ax(axs, fig, 
                                 obs_seq=frames, 
                                 ref_seq=frames, 
                                 matrix=distance_matrix,
                                 title=f"{fig_name}",
                                 seq_cmap=None,
                                 matrix_cmap='gray_r',
                                 rolcol_size=rolcol_size)
    
    plt.tight_layout()

    plt.savefig(save_fig_path)

    plt.close(fig)


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

def create_distance_matrix_video(distance_matrix, frames, save_video_path, fig_name, rolcol_size=2):
    """
    Parameters:
        distance_matrix (Torch.Tensor): the distance between each pair of frames
            shape: (num_frames, num_frames)
        frames (List[PIL.Image]): the frames of the gif file as a list of PIL.Image
    """
    video_writer = imageio.get_writer(save_video_path, fps=15)

    min_dist = distance_matrix.min()
    max_dist = distance_matrix.max()

    for i in range(len(frames)):
        frame = frames[i:i+1]
        distance_row = distance_matrix[i:i+1]

        fig_width = rolcol_size * (len(frames) + 1)
        # Add 1 for the title, 1 for the colorbar, and 4 for the padding
        fig_height = rolcol_size * (1 + 1 + 4)

        fig, axs = plt.subplots(1, 1, figsize=(fig_width, fig_height))

        # For frames that appeared before the current frame, we want to darken the color
        colored_frames = np.array([np.clip(frames[j] * 0.5, 0, 225).astype(np.uint8) if j < i else frames[j] for j in range(len(frames))])

        plot_matrix_as_heatmap_on_ax(axs, fig, 
                                 obs_seq=frame, 
                                 ref_seq=colored_frames, 
                                 matrix=distance_row,
                                 title=f"{fig_name} t={i}",
                                 seq_cmap=None,
                                 matrix_cmap='gray_r',
                                 rolcol_size=rolcol_size,
                                 vmin=min_dist, vmax=max_dist,
                                 matrix_text_font_size=30)
        
        plt.tight_layout()

        video_writer.append_data(np.uint8(fig2img(fig)))

        plt.close(fig)

    video_writer.close()


if __name__ == "__main__":
    gif_path_dict = {
        "door-close": [
            # [Failure] Robot arm barely moves (maybe slightly inching towards the door, but hard to visually see)
            "/share/portal/hw575/CrossQ/train_logs/2024-11-18-195009_sb3_sac_envt=door-close-v2-goal-observable_rm=hand_engineered_nt=ep-len=200/eval/10000_rollouts.gif",
            # [Failure] Robot arm makes more visble intention moving towards the door, but doesn't make contact to the door before the end of the timestep
            "/share/portal/hw575/CrossQ/train_logs/2024-11-18-195009_sb3_sac_envt=door-close-v2-goal-observable_rm=hand_engineered_nt=ep-len=200/eval/40000_rollouts.gif",
            # [Failure] Robot arm moves up instead of towards the door
            "/share/portal/hw575/CrossQ/train_logs/2024-11-18-195009_sb3_sac_envt=door-close-v2-goal-observable_rm=hand_engineered_nt=ep-len=200/eval/60000_rollouts.gif",
            # [Success] Robot arm closes the door then kept moving to its left
            "/share/portal/hw575/CrossQ/train_logs/2024-11-18-195009_sb3_sac_envt=door-close-v2-goal-observable_rm=hand_engineered_nt=ep-len=200/eval/50000_rollouts.gif",
            # [Success] Robot arm closes the door much faster (final rollout after training for 1e6)
            "/share/portal/hw575/CrossQ/train_logs/2024-11-18-195009_sb3_sac_envt=door-close-v2-goal-observable_rm=hand_engineered_nt=ep-len=200/eval/1000000_rollouts.gif"
        ]
    }

    # Set up arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm_model", "-r", type=str, default="dinov2_vitl14_reg", help="VLM model name")
    parser.add_argument("--task_name", "-t", type=str, default="door-close", help="Task name")
    parser.add_argument("--batch_size", "-b", type=int, default=16, help="Batch size for VLM model")
    parser.add_argument("--distance_fn", "-d", type=str, default="cosine", help="Distance function to be used")
    parser.add_argument("--skip_frames", "-s", type=int, default=10, help="Number of frames to skip")

    args = parser.parse_args()

    # Experiment save folder
    EXP_RESULT_FOLDER = "./debugging/vlm_intermediate_distance"

    result_folder = os.path.join(EXP_RESULT_FOLDER, f"{args.task_name}_skip-frames={args.skip_frames}")

    os.makedirs(result_folder, exist_ok=True)

    gif_paths = gif_path_dict[args.task_name]

    preprocess_fn, forward_pass_fn = load_vlm_model_fn(args.vlm_model)

    for i in tqdm(range(len(gif_paths))):
        gif_path = gif_paths[i]

        print(f"    Loading frames from: {gif_path}")
        # Load the frames of the gif file
        frames_tensor, gif_frames = load_frames_to_torch(gif_path, skip_frames=args.skip_frames)

        print(f"    Running {args.vlm_model} on the frames (size={frames_tensor.size()})")
        gif_frames_features = run_vlm_on_gif_path(frames_tensor, preprocess_fn, forward_pass_fn, batch_size=args.batch_size)

        print(f"    Computing the distance matrix between the frames")
        # Compute the distance between each pair of frames
        distance_matrix = compute_frame_to_frame_distance(gif_frames_features, gif_frames_features, DIST_FN_DICT[args.distance_fn])

        # # Plot the distance matrix as a heatmap against the frames
        rollout_timestep = gif_path.split("/")[-1].split("_")[0]
        fig_fname = f"{args.task_name}_t={rollout_timestep}_{args.distance_fn}-distance_matrix.png"
        fig_save_path = os.path.join(result_folder, fig_fname)

        print(f"    Plotted distance matrix to: {fig_save_path}")
        plot_distance_matrix(distance_matrix, np.array(gif_frames), fig_save_path, fig_fname[:-4])

        # video_fname = f"{args.task_name}_t={rollout_timestep}_{args.distance_fn}-distance_video.mp4"
        # video_save_path = os.path.join(result_folder, video_fname)
        # print(f"    Creating distance matrix video to: {video_save_path}")
        # create_distance_matrix_video(distance_matrix, np.array(gif_frames), video_save_path, video_fname[:-4])
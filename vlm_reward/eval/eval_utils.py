from jaxtyping import Float

from typing import List, Union
import torch 
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

def load_gif_frames(path: str, output_type="torch") -> Union[List[Image], Float[torch.Tensor, "n c h w"]]:
    """
    output_type is either "torch" or "pil"

    Load the gif at the path into a torch tensor with shape (frames, channel, height, width)
    """
    gif_obj = Image.open(path)
    frames = [gif_obj.seek(frame_index) or gif_obj.convert("RGB") for frame_index in range(gif_obj.n_frames)]
    if output_type == "pil":
        return frames

    frames_torch = torch.stack([torch.tensor(np.array(frame)).permute(2, 0, 1) for frame in frames])
    return frames_torch

def load_image_from_path(path: str, output_type="torch") -> Union[Image, Float[torch.Tensor, "c h w"]]:
    """
    output_type is either "torch" or "pil"

    Load the image at the path into a torch tensor with shape (channel, height, width)
    """
    frame = Image.open(path).convert("RGB")
    if output_type == "pil":
        return frame

    frame_torch = torch.tensor(np.array(frame)).permute(2,0,1)
    return frame_torch

def load_images_from_paths(paths: List[str], output_type="torch")-> Union[List[Image], Float[torch.Tensor, "n c h w"]]:
    """
    output_type is either "torch" or "pil". If "pil", frames will be loaded to a list of PIL.Image

    Load a batch of images at the given paths into a torch tensor with shape (batch, channel, height, width)
    """
    if output_type == "pil":
        return [load_image_from_path(path, output_type="pil") for path in paths]
    
    return torch.stack([load_image_from_path(path, output_type="torch") for path in paths])

def cut_to_shortest_length(sequences: List):
    """
    cut each array in sequences to the length of the shortest sequence, so they all have the same length
    """
    min_length = min(s.length() for s in sequences)

    return [s[:min_length] for s in sequences]

def pad_to_longest_sequence(sequences):
    """
    pad each array in sequences with global minimum value in sequences, so that they are all the same length, and the range is maintained
    """
    max_len = 0
    min_val = np.inf
    for i in range(len(sequences)):
        max_len = max(len(sequences[i]), max_len)
        min_val = min(min(sequences[i]), min_val)
    for i in range(len(sequences)):
        sequences[i] = np.concatenate((sequences[i], [min_val] * (max_len-len(sequences[i]))))
    return sequences

def rewards_matrix_heatmap(rewards, fp):
    """
    Creates a heatmap of the rewards 
    rewards: np.ndarray, shape (B, N), where N is the rollout length
    """

    ep_rewards = [sum(r) / len(r) for r in rewards]

    rewards = pad_to_longest_sequence(rewards)
    rewards = np.repeat(rewards, 8, axis=0) # repeat for visibility (thickness) in heatmap
    im = plt.imshow(rewards, interpolation='nearest', cmap=plt.cm.hot)
    plt.title(f'Order (best to worst): {np.argsort(ep_rewards)}')
    plt.legend()
    plt.colorbar(im) 

    if not fp.endswith('.png') and not fp.endswith('.jpg'):
        fp = fp + '.png'
    plt.savefig(fp)
    plt.clf()
  
def rewards_line_plot(rewards, labels, fp='outputs/line_plot.png', c=None):
    rewards = np.array(rewards)
    fig, ax = plt.subplots(1)
    for i, gif_rewards in enumerate(rewards):
        avg_cost = np.mean(rewards[i])

        if c is not None:
            ax.plot(range(len(rewards[i])), rewards[i], label=f'{labels[i]}: avg {avg_cost:.3f}', color=c)
        else:
            ax.plot(range(len(rewards[i])), rewards[i], label=f'{labels[i]}: avg {avg_cost:.3f}')

    title = fp.split('/')[-1].split('.')[0]
    ax.set_title(f'Rewards: {title}')
    ax.set_xlabel('Episode step')
    ax.set_ylabel('Reward Score')
    ax.legend()
        
    if not fp.endswith('.png') and not fp.endswith('.jpg'):
        fp = fp + '.png'
        
    plt.savefig(fp)
    plt.clf()
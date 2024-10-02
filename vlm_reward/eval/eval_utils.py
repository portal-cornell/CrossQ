from jaxtyping import Float

from typing import List, Union
import torch 
import numpy as np
import matplotlib.pyplot as plt

import os
import csv

from PIL import Image

def write_metrics_to_csv(metrics, headers, file_name):
    """
    Writes the values from the list metrics to a CSV file with headers.

    :param metrics: List of lists where each sublist represents a row of values to write.
    :param headers: List of strings representing the headers for the CSV.
    :param file_name: String representing the name of the CSV file.
    """
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the headers
        writer.writerow(headers)
        # Write the metrics
        writer.writerows(metrics)

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

def get_filename_from_path(path):    
    return os.path.basename(path).split('.')[0]

def load_np_or_torch_to_torch(path: str):
    """
    loads a .npy file or a .pt file to a torch tensor
    """

    if path.endswith('.pt'):
        return torch.load(path)
    elif path.endswith('.npy'):
        return torch.as_tensor(np.load(path))
    else:
        raise Exception("Unsupported object type")

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

def plot_permutation_diagram(ground_truth_ranking, predicted_ranking, fp, sequence_labels=None):
    # Create a figure and axis
    fig, ax = plt.subplots()

    # order[i] is the index of ranking[i] (i.e., the location in the ranking of the ith element)
    gt_order = torch.empty_like(ground_truth_ranking)
    gt_order[ground_truth_ranking] = torch.arange(len(ground_truth_ranking))

    pred_order = torch.empty_like(predicted_ranking)
    pred_order[predicted_ranking] = torch.arange(len(predicted_ranking))

    # Plot the rankings as scatter points connected by lines
    for i in range(len(gt_order)):
        ax.plot([0, 1], [gt_order[i], pred_order[i]], marker='o')

    # Set the ticks and labels on the x-axis
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Ground Truth', 'Predicted'])

    # Set the y-axis ticks and labels
    ax.set_yticks(np.arange(len(ground_truth_ranking)))
    if sequence_labels is not None:
        ax.set_yticklabels(sequence_labels)
    else:
        ax.set_yticklabels([f'Sequence {i}' for i in range(len(ground_truth_ranking))])

    # Add a grid for better readability
    ax.grid(True)

    # Set labels
    ax.set_xlabel('Rankings')
    ax.set_ylabel('Sequence')
    plt.title('Ranking Comparison')

    plt.tight_layout()

    plt.savefig(fp)
    plt.close()

def create_empty_file(fp):
    open(fp, 'a').close()


def gt_vs_source_heatmap(gt, source, fp):
    # Compute the global min and max for both arrays
    vmin = min(np.min(gt), np.min(source))
    vmax = max(np.max(gt), np.max(source))

    # Create a figure and axis
    fig, ax = plt.subplots(2, figsize=(5, 5))

    # Display heatmap for the first sequence
    heatmap1 = ax[0].imshow([gt], aspect='auto', cmap=plt.cm.hot, interpolation='nearest', vmin=vmin, vmax=vmax)
    ax[0].set_title('Ground Truth')
    ax[0].set_xlabel('Frame')
    ax[0].set_yticks([])  # Remove y-axis ticks for a cleaner look

    # Display heatmap for the second sequence
    heatmap2 = ax[1].imshow([source], aspect='auto', cmap=plt.cm.hot, interpolation='nearest', vmin=vmin, vmax=vmax)
    ax[1].set_title('Model')
    ax[1].set_xlabel('Frame')
    ax[1].set_yticks([])  # Remove y-axis ticks for a cleaner look

    # Add colorbars
    fig.colorbar(heatmap1, ax=ax[0])
    fig.colorbar(heatmap2, ax=ax[1])

    # Display the plot
    plt.tight_layout()
    plt.savefig(fp)
    plt.clf()
    plt.close()

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
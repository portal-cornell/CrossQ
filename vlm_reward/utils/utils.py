import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import functools

from loguru import logger
from tqdm import tqdm
from PIL import Image
import imageio
import io

MINVAL = -0.5

def load_gif_frames(gif_obj):
    frames = [gif_obj.seek(frame_index) or gif_obj.convert("RGB") for frame_index in range(gif_obj.n_frames)]
    return frames

def pad_to_longest_sequence(sequences):
    """
    pad each array in sequences with glboal minimum value in sequences, so that they are all the same length
    """
    max_len = 0
    max_val = - np.inf
    min_val = np.inf
    for i in range(len(sequences)):
        max_len = max(len(sequences[i]), max_len)
        max_val = max(np.max(sequences[i]), max_val)
        min_val = min(np.min(sequences[i]), min_val)
    logger.info(f"\nmax_val={max_val}, min_val={min_val}")
    min_val = MINVAL # TODO: a hack when I know what is the min val (make heatmap looks nicer)
    for i in range(len(sequences)):
        sequences[i] = np.concatenate((sequences[i], [min_val] * (max_len-len(sequences[i]))))
    return sequences

def index_matrix_with_bounds(A, i, j, default):
    """
    Returns A[i,j] if it exists, else default
    """
    if i >= len(A) or j >= len(A[0]) or i < 0 or j < 0:
        return default
    return A[i][j]

def chunk_list(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def pil_to_tensor(frames):
    return torch.stack([torch.tensor(np.array(frame)).permute(2, 0, 1) for frame in frames])


def timing_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        result = func(*args, **kwargs)
        end_event.record()
        
        torch.cuda.synchronize()
        execution_time = start_event.elapsed_time(end_event)
        print(f"Execution time of {func.__name__}: {execution_time / 1000} seconds")
        
        return result
    return wrapper

def create_gif_from_figs(frame_dir, out_path):
    """
    Given a directory containing images, create a gif from them (in the order of their sorted file names)
    """
    images=[]
    image_names = sorted(os.listdir(frame_dir), key=lambda x: int(x.split('.')[0]))

    for filename in image_names:
        image_path = os.path.join(frame_path, filename)
        images.append(imageio.imread(image_path))
        os.remove(image_path)
    
    os.removedirs(frame_dir)
    imageio.mimsave(out_path, images)


def plot_standardized_rewards(rewards, prompts, fp='diff.png'):
    standardized_rewards = (rewards - np.expand_dims(rewards.mean(axis=1), axis=1)) / np.expand_dims(rewards.std(axis=1), axis=1)
    fig, (ax1, ax2) = plt.subplots(1,2)

    for i, gif_rewards in enumerate(standardized_rewards):
        ax1.plot(range(len(rewards[i])), rewards[i], label=f'{prompts[i]}')

        ax2.plot(range(len(gif_rewards)), gif_rewards, label=f'{prompts[i]}')
    
    ax1.set_title('Rewards')
    ax1.set_xlabel('Episode step')
    ax1.set_ylabel('Reward Score')

    ax2.set_title('Standardized Rewards')
    ax2.set_xlabel('Episode step')
    ax2.set_ylabel('Standardized Reward Score')

    ax2.legend()
    plt.gcf().set_size_inches(10, 5)
    plt.savefig(fp, dpi=200)
    plt.clf()

def rewards_matrix_heatmap(rewards, fp):

    ep_rewards = [sum(r) / len(r) for r in rewards]
    print(ep_rewards)
    rewards = pad_to_longest_sequence(rewards)
    rewards = np.repeat(rewards, 8, axis=0) # repeat for visibility (thickness) in heatmap
    # The highest reward should be 0 at a timepoint (meaning that the distance is at 0)
    # The lowest reward should be -1 at a timepoint (meaning that the distance is at 1)
    im = plt.imshow(rewards, interpolation='nearest', vmin=MINVAL, vmax=0, cmap=plt.cm.hot)
    # im = plt.imshow(rewards, interpolation='nearest', cmap=plt.cm.hot)
    # fp[18:-4] is a hack to print out the filename (what axis we are testing)
    plt.title(f'{fp[18:-4]}\nOrder (best to worst): {np.argsort(ep_rewards)[::-1]}') # Sort highest to lowest
    plt.legend()
    plt.colorbar(im) 

    if not fp.endswith('.png') and not fp.endswith('.jpg'):
        fp = fp + '.png'
    plt.savefig(fp)
    plt.clf()

def patch_matching_gif(reward_model, gif_path, target_img_path, best_match_list, cost_list, label):
    # Load the gif
    logger.info(f"Loading the gif at: {gif_path}")
    gif_obj = Image.open(gif_path)
    frames = load_gif_frames(gif_obj)
    frames_transformed = reward_model.prepare_images_parallel(frames)
    logger.debug(f"frames={len(frames)}, {frames[0].size}, frames_transformed={frames_transformed.size()}")

    # Load the target image
    logger.info(f"Loading the target image at: {target_img_path}")
    # transformed_image, grid_size, resize_scale
    target_img, _, _ = reward_model.load_and_prepare_images_parallel(target_img_path)

    gif_writer = imageio.get_writer("axis_exp/patch_matching_gifs/testing.gif", mode="I")
    # Visualize the patches
    for i in tqdm(range(len(frames))):
        if i == len(frames):
            # Don't need to compute the cost for the last frame
            best_match = []
            cost = []
        else:
            best_match = best_match_list[i-1]
            cost =  cost_list[i-1]
        gif_writer.append_data(plot_patch_matching_one_frame(frames_transformed[i], target_img, best_match_list[i-1], cost_list[i-1]))

    # Plot the match where best match list associate the patches and cost list associate the 
    gif_writer.close()

def plot_patch_matching_one_frame(source_img, target_img, best_match_matrix, cost_matrix):
    fig = plt.figure(figsize=(20,10))

    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.imshow(source_img.permute(1, 2, 0))
    ax2.imshow(target_img.permute(1, 2, 0))

    image_size = source_img.size()[1]
    patch_size = 14

    # Calculate number of patches
    num_patches = image_size // patch_size

    # TODO: visulize the patches as grid
    # Add a grid of patch boundaries
    for i in range(num_patches + 1):
        # Vertical line
        ax1.add_line(plt.Line2D((i * patch_size, i * patch_size), (0, image_size), linewidth=1, color='red'))
        # Horizontal line
        ax1.add_line(plt.Line2D((0, image_size), (i * patch_size, i * patch_size), linewidth=1, color='red'))

        # Vertical line
        ax2.add_line(plt.Line2D((i * patch_size, i * patch_size), (0, image_size), linewidth=1, color='red'))
        # Horizontal line
        ax2.add_line(plt.Line2D((0, image_size), (i * patch_size, i * patch_size), linewidth=1, color='red'))

    # Add patches to the images
    # for i in range(num_patches):
    #     for j in range(num_patches):
            # # Create a rectangle patch for ax1
            # rect1 = patches.Rectangle((j*patch_size, i*patch_size), patch_size, patch_size, linewidth=1, edgecolor='r', facecolor='none')
            # ax1.add_patch(rect1)
            
            # # Create a rectangle patch for ax2
            # rect2 = patches.Rectangle((j*patch_size, i*patch_size), patch_size, patch_size, linewidth=1, edgecolor='r', facecolor='none')
            # ax2.add_patch(rect2)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    scatter_frame = Image.open(buf)
    plt.close()

    return scatter_frame

def rewards_line_plot(rewards, labels, fp='outputs/line_plot.png'):
    rewards = np.array(rewards)
    fig, ax = plt.subplots(1)
    for i, gif_rewards in enumerate(rewards):
        avg_cost = np.mean(rewards[i])
        ax.plot(range(len(rewards[i])), rewards[i], label=f'{labels[i]}: avg {avg_cost:.3f}')
    
    ax.set_title('Costs')
    ax.set_xlabel('Episode step')
    ax.set_ylabel('Cost Score')
    ax.legend()
        
    if not fp.endswith('.png') and not fp.endswith('.jpg'):
        fp = fp + '.png'
        
    plt.savefig(fp)
    plt.clf()


def main():
    rewards = np.load('handwritten_prompts/task_muscle_up_simple/rewards_a human doing a muscle-up_muscle_up.npy')
    target_prompts = [            "A picture of a man hanging from a bar",
            "A picture of a man half way through a pull up",
            "A picture of a man finishing a pull up",
            "A picture of a man pushing on top of a bar",
            "A picture of a man finishing a muscle up"]
    plot_standardized_rewards(rewards, target_prompts)

def extended_cosine_similarity(basis1, basis2):
    """
    Inputs are np.ndarrays
    Similarity between vector spaces
    https://www.researchgate.net/publication/350754527_A_measure_for_the_similarity_of_vector_spaces 
    """

    similarities = basis1 @ basis2.T
    d = np.min(np.max(similarities, axis=1), axis=0)
    return d

if __name__=='__main__':
    main()
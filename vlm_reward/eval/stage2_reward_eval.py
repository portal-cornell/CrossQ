import torch

from vlm_reward.eval.model_interface import RewardModel
from jaxtyping import Float
from typing import Callable, List, Tuple, Dict
import scipy

from eval_utils import load_gif_frames, load_image_from_path

from vlm_reward.eval.model_factory import load_reward_model

from omegaconf import DictConfig, OmegaConf
import hydra
import os

def kendalltau_ranking_distance(x_rewards: Float[torch.Tensor, "N"], y_rewards: Float[torch.Tensor, "N"]):
    x_order = torch.argsort(x_rewards)
    y_order = torch.argsort(y_rewards)

    return kendalltau(x_order, y_order)

def kendalltau(x: Float[torch.Tensor, "N"], y: Float[torch.Tensor, "N"]):
    """
    kendall tau distance between two lists represented as pytorch tensors
    """
    x_np = x.numpy()
    y_np = y.numpy()

    return scipy.stats.kendalltau(x_np, y_np)

def kendalltau_average_reward_ranking(x_sequence_rewards, y_sequence_rewards):
    """
    kendall tau distance between the average reward ranking for x_sequence and y_sequence
    i.e., rank the sequences according to their average frame-level reward for x and y,
    and find kt between these two rankings
    """
    x_average_rewards = torch.as_tensor([x.mean() for x in x_sequence_rewards])
    y_average_rewards = torch.as_tensor([y.mean() for y in y_sequence_rewards])

    return kendalltau_ranking_distance(x_average_rewards, y_average_rewards)

def eval_many_sequences(model: RewardModel, 
        source_sequences: List[Float[torch.Tensor, "frames c h w"]],
        target_image: Float[torch.Tensor, "c h w"],
        ground_truth_rewards: List[Float[torch.Tensor, "frames"]],
        evaluation_metric: Callable[[List[Float[torch.Tensor, "frames"]], List[Float[torch.Tensor, "frames"]]], int]):
    """
    Evaluate a list of sequences against ground truth according to evaluation_metric
    Given a list of sequences (gifs), ground truth rewards for each frame in each sequence, and a target image

    TODO: create a ranking of the sequences according to model and ground_truth_rewards and
    evaluate using KT
        - aggregate the rewards for each sequence by summing
    """
    
    model.set_target_embedding(target_image)

    source_rewards = []
    for source_sequence in source_sequences:
        model.set_source_embeddings(source_sequence)
        sequence_rewards = model.predict()
        source_rewards.append(sequence_rewards)

    performance = evaluation_metric(ground_truth_rewards, source_rewards)
    return performance

def eval_sequence(model: RewardModel, 
        source_images: Float[torch.Tensor, "frames c h w"],
        target_image: Float[torch.Tensor, "c h w"],
        ground_truth_rewards: Float[torch.Tensor, "frames"],
        evaluation_metric:  Callable[[List[Float[torch.Tensor, "frames"]], List[Float[torch.Tensor, "frames"]]], int]):
    """
    Evaluate a single sequence of rewards against the ground truth
    
    TODO: use KL or mse
    """
    model.set_target_embedding(target_image)
    model.set_source_embeddings(source_images)

    model_rewards = model.predict()

    performance = evaluation_metric(ground_truth_rewards, model_rewards)
    return performance


def eval_from_paths(
        model: RewardModel,
        source_gif_paths: List[str], 
        target_image_path: str,
        ground_truth_rewards_paths: List[str], 
        evaluation_metric: Callable[[List[Float[torch.Tensor, "frames"]], List[Float[torch.Tensor, "frames"]]], int]):
    """
    source_gif_paths: a list of paths to .gif objects
    ground_truth_rewards_path: a list of tensors, where ground_truth_rewards_path[i] has shape (N,)
        and contains the rewards for each frame in source_gif_paths[i] (a gif of length N) 
    target_image_path: a path to an image object (accessible by PIL.Image.open)
    """

    target_image = load_image_from_path(target_image_path, output_type="torch")

    source_sequences = []
    ground_truth_rewards = []
    for source_gif_path, ground_truth_rewards_path in zip(source_gif_paths, ground_truth_rewards_paths):
        source_frames = load_gif_frames(source_gif_path, output_type="torch")
        sequence_ground_truth_rewards = torch.load(ground_truth_rewards_path)

        source_sequences.append(source_frames)
        ground_truth_rewards.append(sequence_ground_truth_rewards)

    performance = eval_many_sequences(model, source_sequences, target_image, ground_truth_rewards, evaluation_metric)
    return performance

def match_rewards_to_sources(source_filenames, reward_filenames):
    """
    Assumes there is a 1:1 matching between source filenames and reward_filenames, but they are both unordered
    Then, once they are sorted, source_filenames[i] is guaranteed to correspond to reward_filenames[i]
    """
    return sorted(source_filenames), sorted(reward_filenames)

def get_matched_source_and_reward_files(source_sequence_dir, reward_dir) -> Tuple[List[str], List[str]]:
    """
    Returns (source_paths, reward_paths), such that reward_paths[i] contains rewards corresponding to source_paths[i]
    
    Assumes there is a 1:1 matching between source filenames and reward_filenames, but they are both unordered
    Then, once they are sorted, source_filenames[i] is guaranteed to correspond to reward_filenames[i]
    """

    source_filenames = sorted(os.listdir(source_sequence_dir))
    reward_filenames = sorted(os.listdir(reward_dir))

    source_paths = [os.path.join(source_sequence_dir, name) for name in source_filenames]
    reward_paths = [os.path.join(reward_dir, name) for name in reward_filenames]

    return source_paths, reward_paths

def eval_from_config(config: DictConfig, metric_enum: Dict[str, Callable]):
    model_config_dict = OmegaConf.to_container(config.reward_model, resolve=True, throw_on_missing=True)

    reward_model = load_reward_model(rank=0, 
                                    worker_actual_batch_size=config.reward_model.reward_batch_size,  
                                    model_name=config.reward_model.vlm_model, 
                                    model_config_dict=model_config_dict)

    source_paths, reward_paths = get_matched_source_and_reward_files(config.eval_data.source_sequence_dir, config.eval_data.reward_dir)

    # use the eval_data target path instead of reward_model.pos_image_paths
    # this is because the path is generally specific to the set of data that is collected, and will change here
    target_path = config.eval_data.target_image_path
    
    eval_metric = metric_enum[config.eval_metric]
    eval_from_paths(reward_model, source_paths, target_path, reward_paths, eval_metric)

@hydra.main(version_base=None, config_path="configs", config_name="eval_config")
def main(cfg: DictConfig):

    metric_enum = {
        "kendall-tau": kendalltau_ranking_distance,
        "kendalltau-average-reward-ranking": kendalltau_average_reward_ranking,
    }

    eval_from_config(cfg, metric_enum)

if __name__=="__main__":
    main()
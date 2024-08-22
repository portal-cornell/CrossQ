import torch
import numpy as np
from vlm_reward.eval.model_interface import RewardModel
from jaxtyping import Float
from typing import Callable, List, Tuple, Dict, Optional
import scipy

from eval_utils import load_gif_frames, load_image_from_path, load_np_or_torch_to_torch, plot_permutation_diagram, gt_vs_source_heatmap, get_filename_from_path, create_empty_file
from data_parse import parse_mujoco_eval_dir
from vlm_reward.eval.model_factory import load_reward_model

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
import os

from loguru import logger

def kendalltau_ranking_distance(x_rewards: Float[torch.Tensor, "N"], 
                                y_rewards: Float[torch.Tensor, "N"]) -> float:
    x_order = torch.argsort(x_rewards)
    y_order = torch.argsort(y_rewards)

    return kendalltau(x_order, y_order)

def kendalltau(x: Float[torch.Tensor, "N"], y: Float[torch.Tensor, "N"]):
    """
    kendall tau distance between two lists represented as pytorch tensors
    """
    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()

    return scipy.stats.kendalltau(x_np, y_np)

def kendalltau_average_reward_ranking(x_sequence_rewards: List[Float[torch.Tensor, "frames"]], 
                                    y_sequence_rewards: List[Float[torch.Tensor, "frames"]]) -> float:
    """
    kendall tau distance between the average reward ranking for x_sequence and y_sequence
    i.e., rank the sequences according to their average frame-level reward for x and y,
    and find kt between these two rankings
    """
    x_average_rewards = torch.as_tensor([x.mean() for x in x_sequence_rewards])
    y_average_rewards = torch.as_tensor([y.mean() for y in y_sequence_rewards])
    return kendalltau_ranking_distance(x_average_rewards, y_average_rewards)

def plot_sequence_rewards_ranking_permutation(x_sequence_rewards, y_sequence_rewards, fp):
    x_average_ranking = torch.argsort(torch.as_tensor([x.mean() for x in x_sequence_rewards]))
    y_average_ranking = torch.argsort(torch.as_tensor([y.mean() for y in y_sequence_rewards]))
    plot_permutation_diagram(x_average_ranking, y_average_ranking, fp)

def eval_sequence(model: RewardModel, 
        source_images: Float[torch.Tensor, "frames c h w"],
        target_image: Float[torch.Tensor, "c h w"],
        ground_truth_rewards: Float[torch.Tensor, "frames"],
        evaluation_metric:  Callable[[List[Float[torch.Tensor, "frames"]], List[Float[torch.Tensor, "frames"]]], float]):
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
        target_image_path: str,
        sources_and_rewards: List[Tuple[str, str]], 
        cross_sequence_evaluation_metric: Callable[[List[Float[torch.Tensor, "frames"]], List[Float[torch.Tensor, "frames"]]], float],
        within_sequence_evaluation_metric: Optional[Callable[[Float[torch.Tensor, "frames"], Float[torch.Tensor, "frames"]], float]] = None,
        cross_sequence_plotter: Optional[Callable[[List[Float[torch.Tensor, "frames"]], List[Float[torch.Tensor, "frames"]], str], None]] = None,
        within_sequence_plotter: Optional[Callable[[Float[torch.Tensor, "frames"], Float[torch.Tensor, "frames"], str], None]] = None):
    
    """
    Evaluate a list of sequences against ground truth according to evaluation_metric
    
    sources_and_rewards: a list of tuples containing a gif and the rewards for each frame in that gif
    target_image_path: a path to an image object (accessible by PIL.Image.open)
    cross_sequence_evaluation_metric: the function that evaluates the performance w.r.t ground truth across all sequences
    within_sequence_evaluation_metric: (optional) the function that evaluates the performance w.r.t ground truth within a given sequence
    cross_sequence_plotter: (optional) the function that visualizes the differences between the ground truth and model across sequences. Takes an output path as a parameter
    within_sequence_plotter: (optional) the function that visualizes the differences between the ground truth and model within sequences. Takes an output path as a parameter
    """

    target_image = load_image_from_path(target_image_path, output_type="torch")
    model.set_target_embedding(target_image)

    source_rewards = []
    ground_truth_rewards = []
    within_sequence_performances = []
    data_save_dir = HydraConfig.get().runtime.output_dir

    for source_gif_path, ground_truth_rewards_path in sources_and_rewards:
        # cut off the last frame and reward for all sequences, because it often has weird stuff
        # (ex. text on the screen, reset robot position)
        # TODO: this is pretty hacky rn
        source_frames = load_gif_frames(source_gif_path, output_type="torch")[:-1]
        sequence_ground_truth_rewards = load_np_or_torch_to_torch(ground_truth_rewards_path)[:-1]

        model.set_source_embeddings(source_frames)
        sequence_rewards = model.predict()

        source_rewards.append(sequence_rewards)
        ground_truth_rewards.append(sequence_ground_truth_rewards)

        # perform within sequence evaluation and saving
        sequence_name = get_filename_from_path(source_gif_path)
        sequence_dir = f"{data_save_dir}/{sequence_name}"
        os.makedirs(sequence_dir)
        
        # create a link to the original sequence
        os.link(source_gif_path, f'{sequence_dir}/sequence.gif')

        # evaluate within the sequence, and save the metric to an empty file
        if within_sequence_evaluation_metric is not None:
            within_sequence_performance = within_sequence_evaluation_metric(sequence_ground_truth_rewards, sequence_rewards)
            within_sequence_performances.append(within_sequence_performance)
            create_empty_file(f"{sequence_dir}/performance={within_sequence_performance}")

        # plot within the sequence
        if within_sequence_plotter is not None:
            within_sequence_plotter(sequence_ground_truth_rewards, sequence_rewards, f"{sequence_dir}/within_sequence_rewards.png")

        # save a tensor of rewards and ground truth rewards
        torch.save(sequence_rewards, f"{sequence_dir}/model_rewards.pt")
        torch.save(torch.as_tensor(sequence_ground_truth_rewards), f"{sequence_dir}/gt_rewards.pt")

    cross_sequence_performance = cross_sequence_evaluation_metric(source_rewards, ground_truth_rewards)
    create_empty_file(f"{data_save_dir}/performance={within_sequence_performance}")

    cross_sequence_plotter(source_rewards, ground_truth_rewards, f"{data_save_dir}/cross_sequence_rewards.png")

    logger.info(f"Cross sequence performance: {cross_sequence_performance}")
    logger.info(f"Average within sequence performance: {within_sequence_performance}")

def eval_from_config(config: DictConfig):
    model_config_dict = OmegaConf.to_container(config.reward_model, resolve=True, throw_on_missing=True)
    
    sources_and_rewards = parse_mujoco_eval_dir(config.eval_data.sequence_and_reward_dir, get_every_nth=10)

    reward_model = load_reward_model(rank=0, 
                                    worker_actual_batch_size=config.reward_model.reward_batch_size,  
                                    model_name=config.reward_model.vlm_model, 
                                    model_config_dict=model_config_dict)

    target_path = config.reward_model.pos_image_path[0] # assume just 1 target image path for now
    
    within_sequence_evaluation_metric = kendalltau_ranking_distance
    cross_sequence_evaluation_metric = kendalltau_average_reward_ranking

    cross_sequence_plotter = plot_sequence_rewards_ranking_permutation
    # plot heatmap with ground truth on top, preds on bottom
    within_sequence_plotter = lambda gt, preds, fp: gt_vs_source_heatmap(gt.cpu().numpy(), preds.cpu().numpy(), fp) 
    
    eval_from_paths(model=reward_model, 
                    target_image_path=target_path, 
                    sources_and_rewards=sources_and_rewards, 
                    cross_sequence_evaluation_metric=cross_sequence_evaluation_metric, 
                    within_sequence_evaluation_metric=within_sequence_evaluation_metric,
                    cross_sequence_plotter=cross_sequence_plotter,
                    within_sequence_plotter=within_sequence_plotter)

@hydra.main(version_base=None, config_path="configs", config_name="eval_config")
def main(cfg: DictConfig):

    eval_from_config(cfg)

if __name__=="__main__":
    main()
import torch
import torch.nn.functional as F
import numpy as np
import scipy
import os
from loguru import logger

from vlm_reward.reward_models.model_interface import RewardModel
from vlm_reward.reward_models.model_factory import load_reward_model
from vlm_reward.eval.eval_utils import load_gif_frames, load_image_from_path, load_np_or_torch_to_torch, plot_permutation_diagram, gt_vs_source_heatmap,get_filename_from_path, create_empty_file, write_metrics_to_csv
from vlm_reward.eval.data_parse import parse_mujoco_eval_dir

from jaxtyping import Float
from typing import Callable, List, Tuple, Dict, Optional

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig

def kendalltau(x: Float[torch.Tensor, "N"], y: Float[torch.Tensor, "N"]) -> float:
    """
    kendall tau distance between two lists represented as pytorch tensors
    """
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    return scipy.stats.kendalltau(x_np, y_np)[0] # only get the statistic, not the p value

def z_score_l2(x: Float[torch.Tensor, "N"], y: Float[torch.Tensor, "N"]) -> float:
    """
    l2 distance between the z scores of the input tensors
    """
    x_z_scores = (x - x.mean()) / x.std()
    y_z_scores = (y - y.mean()) / y.std()

    return torch.linalg.vector_norm(x_z_scores.detach().cpu() - y_z_scores.detach().cpu())

def spearman(x: Float[torch.Tensor, "N"], y: Float[torch.Tensor, "N"]) -> float:
    """
    spearman rank correlation coefficient between two tensors (measure of monotonic relationship)
    i.e., does y go in the same direction as x
    """
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    return scipy.stats.spearmanr(x_np, y_np)[0] # only get the statistic, not the p value

def spearman_average_reward(x_sequence_rewards: List[Float[torch.Tensor, "frames"]], 
                                    y_sequence_rewards: List[Float[torch.Tensor, "frames"]]) -> float:
    """
    spearman correlation between average rewards for x and y
    """
    x_average_rewards = torch.as_tensor([x.mean() for x in x_sequence_rewards])
    y_average_rewards = torch.as_tensor([y.mean() for y in y_sequence_rewards])
    return spearman(x_average_rewards, y_average_rewards)

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

def plot_sequence_rewards_ranking_permutation(x_sequence_rewards, y_sequence_rewards, fp, sequence_labels=None):
    x_average_ranking = torch.argsort(torch.as_tensor([x.mean() for x in x_sequence_rewards]))
    
    if sequence_labels is not None:
        sequence_labels = [sequence_labels[rank] for rank in x_average_ranking] # permute the labels to match the ranking

    y_average_ranking = torch.argsort(torch.as_tensor([y.mean() for y in y_sequence_rewards]))

    plot_permutation_diagram(x_average_ranking, y_average_ranking, fp, sequence_labels=sequence_labels)

def eval_from_paths(
        model: RewardModel,
        target_image_path: str,
        sources_and_rewards: List[Tuple[str, str]], 
        image_transform: Optional[Callable[[Float[torch.Tensor, "frames c h w"]], Float[torch.Tensor, "frames c h w"]]] = None,
        cross_sequence_evaluation_metric: Optional[Callable[[List[Float[torch.Tensor, "frames"]], List[Float[torch.Tensor, "frames"]]], float]] = None,
        within_sequence_evaluation_metric: Optional[Callable[[Float[torch.Tensor, "frames"], Float[torch.Tensor, "frames"]], float]] = None,
        cross_sequence_plotter: Optional[Callable[[List[Float[torch.Tensor, "frames"]], List[Float[torch.Tensor, "frames"]], str], None]] = None,
        within_sequence_plotter: Optional[Callable[[Float[torch.Tensor, "frames"], Float[torch.Tensor, "frames"], str], None]] = None):
    
    """
    Evaluate a list of sequences against ground truth according to evaluation_metric
    
    sources_and_rewards: a list of tuples containing a gif and the rewards for each frame in that gif
    target_image_path: a path to an image object (accessible by PIL.Image.open)
    image_transform: the transform to apply after images are loaded (ex. a crop or resize)
    cross_sequence_evaluation_metric: the function that evaluates the performance w.r.t ground truth across all sequences
    within_sequence_evaluation_metric: (optional) the function that evaluates the performance w.r.t ground truth within a given sequence
    cross_sequence_plotter: (optional) the function that visualizes the differences between the ground truth and model across sequences. Takes an output path as a parameter
    within_sequence_plotter: (optional) the function that visualizes the differences between the ground truth and model within sequences. Takes an output path as a parameter
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    source_rewards = []
    ground_truth_rewards = []
    within_sequence_performances = []
    times = []
    data_save_dir = HydraConfig.get().runtime.output_dir

    for source_gif_path, ground_truth_rewards_path in sources_and_rewards[:2]:
        # cut off the last frame and reward for all sequences, because it often has weird stuff
        # (ex. text on the screen, reset robot position)
        # TODO: this is pretty hacky rn

        source_frames_raw = load_gif_frames(source_gif_path, output_type="torch")[:-1]
        source_frames = image_transform(source_frames_raw)
        sequence_ground_truth_rewards = load_np_or_torch_to_torch(ground_truth_rewards_path)[:-1]

        start.record()

        model.set_source_embeddings(source_frames)
        sequence_rewards = model.predict()

        end.record()
        torch.cuda.synchronize() 
        elapsed_time_s = start.elapsed_time(end) / 1000
        times.append(elapsed_time_s) # save the time per sequence

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

        # plot within the sequence
        if within_sequence_plotter is not None:
            within_sequence_plotter(sequence_ground_truth_rewards, sequence_rewards, f"{sequence_dir}/within_sequence_rewards.png")

        # save a tensor of rewards and ground truth rewards
        torch.save(sequence_rewards, f"{sequence_dir}/model_rewards.pt")
        torch.save(torch.as_tensor(sequence_ground_truth_rewards), f"{sequence_dir}/gt_rewards.pt")
        logger.info(f"Inference completed in: {elapsed_time_s:.3f}s")

    cross_sequence_performance = cross_sequence_evaluation_metric(ground_truth_rewards, source_rewards)
    avg_within_sequence_performance = torch.as_tensor(within_sequence_performances).mean().item()
    max_within_sequence_performance = torch.as_tensor(within_sequence_performances).max().item()
    min_within_sequence_performance = torch.as_tensor(within_sequence_performances).min().item()
    avg_time_per_frame = sum(times) / len(times)

    write_metrics_to_csv(metrics = [[
                            cross_sequence_performance, 
                            avg_within_sequence_performance, 
                            max_within_sequence_performance, 
                            min_within_sequence_performance, 
                            avg_time_per_frame]], 
                        headers = [
                            f"Cross sequence {cross_sequence_evaluation_metric.__name__}",
                            f"Average within sequence {within_sequence_evaluation_metric.__name__}", 
                            f"Max within sequence {within_sequence_evaluation_metric.__name__}", 
                            f"Min within sequence {within_sequence_evaluation_metric.__name__}", 
                            "Average time per gif (s)"],
                        file_name = f"{data_save_dir}/performance.csv")

    cross_sequence_plotter(ground_truth_rewards, source_rewards, f"{data_save_dir}/cross_sequence_rewards.png")

    logger.info(f"Cross sequence performance: {cross_sequence_performance}")
    logger.info(f"Average within sequence performance: {avg_within_sequence_performance}")
    logger.info(f"Average time per frame: {avg_time_per_frame}")

def eval_from_config(config: DictConfig):
    model_config_dict = OmegaConf.to_container(config.reward_model, resolve=True, throw_on_missing=True)
    
    sources_and_rewards = parse_mujoco_eval_dir(config.eval_data.sequence_and_reward_dir, get_every_nth=5)

    reward_model = load_reward_model(rank=0, 
                                    worker_actual_batch_size=config.reward_model.reward_batch_size,  
                                    model_name=config.reward_model.name, 
                                    model_config_dict=model_config_dict)
    reward_model.eval()

    target_path = config.reward_model.pos_image_path[0] # assume just 1 target image path for now
    
    image_transform = lambda image: F.interpolate(image, size=(224,224), mode="bilinear")
    
    # don't use spearman within sequence, because we are more concerned with the large changes across phases
    # in the sequence, not small changes that may occur to the order (due to the models being less stable within the sequence)
    within_sequence_evaluation_metric = z_score_l2 
    cross_sequence_evaluation_metric = spearman_average_reward

    source_names = [os.path.basename(path_pair[0]) for path_pair in sources_and_rewards]
    cross_sequence_plotter = lambda x, y, fp: plot_sequence_rewards_ranking_permutation(x, y, fp, sequence_labels=source_names)
    # plot heatmap with ground truth on top, preds on bottom
    within_sequence_plotter = lambda gt, preds, fp: gt_vs_source_heatmap(gt.cpu().detach().numpy(), preds.cpu().detach().numpy(), fp) 
    
    eval_from_paths(model=reward_model, 
                    target_image_path=target_path, 
                    sources_and_rewards=sources_and_rewards, 
                    image_transform=image_transform,
                    cross_sequence_evaluation_metric=cross_sequence_evaluation_metric, 
                    within_sequence_evaluation_metric=within_sequence_evaluation_metric,
                    cross_sequence_plotter=cross_sequence_plotter,
                    within_sequence_plotter=within_sequence_plotter)

@hydra.main(version_base=None, config_path="configs", config_name="eval_config")
def main(cfg: DictConfig):

    eval_from_config(cfg)

if __name__=="__main__":
    main()
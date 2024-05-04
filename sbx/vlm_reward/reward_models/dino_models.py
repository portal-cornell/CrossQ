
import os
import numpy as np
import concurrent
from functools import lru_cache
from collections import defaultdict

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
import ot

import cv2
from PIL import Image
import imageio

from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

import matplotlib
from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import index_matrix_with_bounds, create_gif_from_figs, timing_decorator


import fastdtw

class Dino2FeatureExtractor:
    """
    Get patch level features from a DinoV2 model
    Crops input images to the correct size, and handles resizing projections later
    Adopted from: https://github.com/facebookresearch/dinov2/blob/255861375864acdd830f99fdae3d9db65623dafe/notebooks/features.ipynb
    """

    def __init__(self, 
                model_name="dinov2_vitb14", 
                repo_name="facebookresearch/dinov2", 
                edge_size=448, 
                half_precision=False,
                device="cuda"):
        self.repo_name = repo_name
        self.model_name = model_name
        self.edge_size = edge_size
        self.half_precision = half_precision
        self.device = device

        if self.half_precision:
            self.model = torch.hub.load(repo_or_dir=repo_name, model=model_name).half().to(self.device)
        else:
            self.model = torch.hub.load(repo_or_dir=repo_name, model=model_name).to(self.device)

        self.model.eval()

        # ensure images can evenly be broken into patches after applying self.transform
        assert self.edge_size % self.model.patch_size == 0

        self.transform = transforms.Compose([
            transforms.Resize(size=edge_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(size=edge_size) # convert to square
        ])
                
        self.normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) # imagenet defaults

    def extract_features(self, images_tensor):
        with torch.inference_mode():
            images_tensor = self.normalize(images_tensor)
            
            if len(images_tensor.shape) == 3:
                image_batch = images_tensor.unsqueeze(0)
            else:
                image_batch = images_tensor
            
            if self.half_precision:
                image_batch = image_batch.half()
            else:
                image_batch = image_batch

            image_batch = image_batch.to(self.device)
            
            all_tokens = self.model.get_intermediate_layers(image_batch)
            
            tokens = all_tokens[0].squeeze()
        return tokens

    def prepare_images_parallel(self, images):
        """
        images: list of PIL.Images or Torch.Tensor
        Applies self.transform on the given batch of images using multithreading
        """
        # If a list of images is provided, use multi-threading for parallel processing
        if isinstance(images, list):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                
                futures = executor.map(self._prepare_single_image, images)
                processed_images = [result[0] for result in futures]
            return torch.stack(processed_images)
        else:  # Single image case
            return self._prepare_single_image(images)


    def load_and_prepare_images_parallel(self, paths):
        """
        paths: list of paths to images
        caches last three inputs, in case you call it on targets, then source, then targets again
        """
        # If a list of images is provided, use multi-threading for parallel processing
        if isinstance(paths, list):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                
                futures = executor.map(self._load_and_prepare_single_image, paths)
                processed_images = [result[0] for result in futures]
            
            return torch.stack(processed_images)
        else:  # Single image case
            return self._load_and_prepare_single_image(paths)

    @lru_cache(maxsize=3)
    def _load_and_prepare_single_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        return self._prepare_single_image(image)

    
    def _prepare_single_image(self, image):
        """
        image is a torch.Tensor or PIL.Image
        """
        if isinstance(image, Image.Image):
            image = transforms.functional.to_tensor(image)
            
        transformed_image = self.transform(image)

        resize_scale = image.shape[2] / transformed_image.shape[2]
        grid_size = self.get_grid_size(transformed_image)

        return transformed_image, grid_size, resize_scale

    def prepare_images_linear(self, images):
        processed_images = []
        for img in images:
            processed = self._prepare_single_image(img)[0]
            processed_images.append(processed)
        return torch.stack(processed_images).to(self.device)

    def get_grid_size(self, img):
        C, H, W = img.shape

        return (H // self.model.patch_size, W // self.model.patch_size)

    def prepare_mask(self, mask_image_numpy, grid_size, resize_scale):
        cropped_mask_image_numpy = mask_image_numpy[:int(grid_size[0]*self.model.patch_size*resize_scale), :int(grid_size[1]*self.model.patch_size*resize_scale)]

        image = Image.fromarray(cropped_mask_image_numpy)
        resized_mask = image.resize((grid_size[1], grid_size[0]), resample=Image.Resampling.NEAREST)
        resized_mask = np.asarray(resized_mask).flatten()
        return resized_mask
    

    def idx_to_source_position(self, idx, grid_size, resize_scale):
        row = (idx // grid_size[1])*self.model.patch_size*resize_scale + self.model.patch_size / 2
        col = (idx % grid_size[1])*self.model.patch_size*resize_scale + self.model.patch_size / 2
        return row, col
    
    def get_embedding_visualization(self, tokens, grid_size, resized_mask=None):
        pca = PCA(n_components=3)
        if resized_mask is not None:
            tokens = tokens[resized_mask]

        reduced_tokens = pca.fit_transform(tokens.astype(np.float32))

        if resized_mask is not None:
            tmp_tokens = np.zeros((*resized_mask.shape, 3), dtype=reduced_tokens.dtype)
            tmp_tokens[resized_mask] = reduced_tokens
            reduced_tokens = tmp_tokens

        reduced_tokens = reduced_tokens.reshape((*grid_size, -1))
        normalized_tokens = (reduced_tokens-np.min(reduced_tokens))/(np.max(reduced_tokens)-np.min(reduced_tokens))
        return normalized_tokens


class Image2ImageMetric(nn.Module):
    def __init__(self, 
                feature_extractor, 
                patch_masker, 
                use_patch_mask = True,
                device='cuda',
                **kwargs):
        """
        Compute distances between a target image and set of source images in dino feature space

        feature_extractor: a model that extracts patch-level features from a given image
        patch_masker: a model that returns relevant patches of a given image
        """
        super().__init__(**kwargs)
        self.feature_extractor = feature_extractor
        self.patch_masker = patch_masker
        self.use_patch_mask = use_patch_mask
        self.device = device

    def update_target_image(self, target_image, mask_thresh=.5):
        target_tensor, target_human_mask, target_features_masked, masked_target_feature_to_feature, target_grid_size, target_resize_scale = self.extract_features_and_scales(target_image, mask_thresh=mask_thresh)

        self.target_features_masked = target_features_masked
        self.target_human_mask = target_human_mask
        self.masked_target_features_to_feature = masked_target_feature_to_feature
        self.target_grid_size = target_grid_size
        self.target_resize_scale = target_resize_scale

    def extract_masked_features_batched(self, tensors, mask_thresh=.5, batch_size=16):
        """
        Split input tensor into batches, and then extract masked features
        tensors: (N, 3, H, W), where N is the number of images
        """
        if batch_size > 1:
            batch_tensors = torch.split(tensors, batch_size)
        else:
            batch_tensors = tensors

        all_features = torch.as_tensor([], device = self.device)
        all_masks = torch.as_tensor([], dtype=torch.bool, device=self.device)


        for batch in batch_tensors:
            features, feature_level_human_mask = self.extract_masked_features(batch, mask_thresh)
            all_features = torch.concat((all_features, features))
            all_masks = torch.concat((all_masks, feature_level_human_mask))

        return all_features, all_masks

    def extract_masked_features(self, batch, mask_thresh=.5):
        """
        Extract masked features from a batch
        batch: (B, 3, H, W), where B is number of images in batch
        """
        features = self.feature_extractor.extract_features(batch)

        _, n_features, _ = features.shape
        feature_indices = np.arange(n_features)
        grid_size = self.feature_extractor.get_grid_size(batch[0])
        rows, columns = self.feature_extractor.idx_to_source_position(feature_indices, grid_size, 1)

        if self.use_patch_mask:
            human_masks = self.patch_masker(batch, thresh=mask_thresh)[:,0,...]
        else:
            human_masks = torch.ones_like(batch)[:,0,...].bool()

        feature_level_human_mask = human_masks[:, rows, columns].bool()

        return features, feature_level_human_mask

    def prepare_images(self, images):
        return self.feature_extractor.prepare_images_parallel(images)

    def extract_features_and_scales(self, image, mask_thresh):
        """
        **DEPRECATED** (only here for compatibility with past experiments)
        Given an image and a human segmentation mask, obtain masked DINO features, and return scales and grid for patches
        """
        tensor, grid_size, resize_scale = self.feature_extractor._prepare_single_image(image)
        features = self.feature_extractor.extract_features(tensor)

        if self.use_patch_mask:
            human_mask = self.patch_masker(tensor, thresh=mask_thresh)[0][0]
        else:
            human_mask = torch.ones_like(tensor)[0].bool()

        feature_level_human_mask = self.get_feature_level_human_mask(human_mask, features, grid_size)
        features_masked = features[feature_level_human_mask]
        masked_feature_to_feature = torch.nonzero(feature_level_human_mask)
        return tensor, human_mask, features_masked, masked_feature_to_feature, grid_size, resize_scale

    def get_feature_level_human_mask(self, human_mask, features, grid_size):
        """
        **DEPRECATED** (only here for use with extract_features_and_scales)
        """
        feature_level_human_mask = torch.zeros(len(features))
        for idx in range(len(features)):
            r, c = self.feature_extractor.idx_to_source_position(idx, grid_size, 1)
            feature_level_human_mask[idx] = human_mask[int(r), int(c)]
        return feature_level_human_mask.bool()



class PatchWassersteinDistance(Image2ImageMetric):

    def __init__(self, 
                feature_extractor, 
                patch_masker):

        super().__init__(feature_extractor, patch_masker)
  
    def forward(self, source_tensors, target_tensors, source_threshold=.001, target_threshold=.5):
        """
        Compute wasserstein distance between a batch of source frames and target frames
        Returns [d(source_frames[i], target_frames[i]) for all i]
        source_threshold and target_threshold are used for the human masking model
        """

        source_features, source_masks = self.extract_masked_features(source_tensors, mask_thresh=source_threshold)
        target_features, target_masks = self.extract_masked_features(target_tensors, mask_thresh=target_threshold)
        return self.compute_distance_parallel(source_features, source_masks, target_features, target_masks)

  
    def forward_batched(self, source_tensors, target_tensors, source_threshold=.001, target_threshold=.5, batch_size=16):
        """
        Compute wasserstein distance between a set of source frames and target frames by breaking them into batches
        Returns [d(source_frames[i], target_frames[i]) for all i]
        source_threshold and target_threshold are used for the human masking model
        """

        source_features, source_masks = self.extract_masked_features_batched(source_tensors, mask_thresh=source_threshold, batch_size=batch_size)
        target_features, target_masks = self.extract_masked_features_batched(target_tensors, mask_thresh=target_threshold, batch_size=batch_size)
        return self.compute_distance_parallel(source_features, source_masks, target_features, target_masks)


    def forward_paths(self, source_paths, target_paths, source_threshold=.001, target_threshold=.5, batch_size=16):
        
        source_tensors = self.feature_extractor.load_and_prepare_images_parallel(source_paths).to(self.device)
        target_tensors = self.feature_extractor.load_and_prepare_images_parallel(target_paths).to(self.device)
        return self.forward_batched(source_tensors, target_tensors, source_threshold, target_threshold, batch_size)

    @timing_decorator
    def forward_cached_target(self, frames, human_threshold, d='euclidean', batch_size=16):
        """
        NOTE: INCOMPLETE
        Obtain wasserstein distances between all frames and the current target frame
        Allows for caching of the target frame features
        human_threshold is used to mask the input frames
        """
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        frame_tensors = self.feature_extractor.prepare_images_parallel(frames).to(self.device)
        end_event.record()
        
        ### timing
        torch.cuda.synchronize()
        execution_time = start_event.elapsed_time(end_event)
        print(f"Image loading time: {execution_time / 1000} seconds")
        ####

        start_event.record()
        all_features, all_masks = self.extract_masked_features_batched(frame_tensors, mask_thresh=human_threshold, batch_size=batch_size)
        end_event.record()

        ### timing
        torch.cuda.synchronize()
        execution_time = start_event.elapsed_time(end_event)
        print(f"Feature extraction time: {execution_time / 1000} seconds")
        ###
        
        start_event.record()

        target_features_masked_np = self.target_features_masked.cpu().numpy()

        def compute_wasserstein_source(feature, mask):
            return self.compute_patchwise_wasserstein(feature[mask], target_features_masked_np, d=d)


        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = executor.map(compute_wasserstein_source, all_features.cpu().numpy(), all_masks.cpu().numpy()) 
            wassersteins = [future for future in futures]
        end_event.record()

        ### timing
        torch.cuda.synchronize()
        execution_time = start_event.elapsed_time(end_event)
        print(f"Optimal Transport Calculation: {execution_time / 1000} seconds")
        ###

        return torch.as_tensor(wassersteins, device=self.device)

    def compute_distance_parallel(self, source_features, source_masks, target_features, target_masks):
        """
        Inputs are torch tensors
        """
        
        def compute_wasserstein_given_masks(source_feature, source_mask, target_feature, target_mask):
            """
            Inputs must be numpy arrays
            """
            return self.compute_patchwise_wasserstein(source_feature[source_mask],target_feature[target_mask], d='euclidean')

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = executor.map(compute_wasserstein_given_masks, 
                                    source_features.cpu().numpy(), 
                                    source_masks.cpu().numpy(),
                                    target_features.cpu().numpy(),
                                    target_masks.cpu().numpy()) 
            wassersteins = [future for future in futures]
        
        return wassersteins

    
    @staticmethod
    def compute_patchwise_wasserstein(features, target_features, d='euclidean'):
                        
        match d:
            case 'cosine':
                M = target_features @ features.T 
                M *= 1 / np.tile(np.linalg.norm(target_features, axis=1), (M.shape[1], 1)).T
                M *= 1 / np.tile(np.linalg.norm(features, axis=1), (M.shape[0], 1))
                M *= -1 # turn the similarity into a cost
            case 'euclidean':
                M = cdist(target_features, features)
        
        if M.shape[1] == 0:
            print('Error: no features found. Distance is considered -1. Consider decreasing the human threshold for the source or target')
            return -1
        else:
            # for now, use constant weights
            target_weights = []
            source_weights = []

            # for some reason, ot.sinkhorn2 is much faster on cpu. There must be a better way than this
            wasser= ot.sinkhorn2(target_weights, source_weights, M, reg=1,log=False)
            
        return wasser

    @staticmethod
    def compute_gromov_wasserstein(features, target_features):
        """
        In development...
        """
        C1 = cdist(features, features)
        C2 = cdist(target_features, target_features)
        coupling = ot.gromov.entropic_gromov_wasserstein(C1, C2)
        wasser = coupling @ C2 @ C1

        return wasser

class PatchMeanFeatureDistance(Image2ImageMetric):

    def __init__(self,  
                feature_extractor, 
                patch_masker):

        super().__init__(feature_extractor, patch_masker, target_human_threshold)

    def forward(self, source_frames, target_frames, source_human_threshold, target_human_threshold, d='euclidean'):
        source_features, source_masks = self.extract_masked_features(source_tensors, mask_thresh=source_threshold)
        target_features, target_masks = self.extract_masked_features(target_tensors, mask_thresh=target_threshold)
        return self.compute_mean_distance(source_features, source_masks, target_features, target_masks, d)

    def forward_batched(self, source_frames, target_frames, source_human_threshold, target_human_threshold, d='euclidean', batch_size=16):        
        source_features, source_masks = self.extract_masked_features_batched(frame_tensors, mask_thresh=source_human_threshold, batch_size=batch_size)
        target_features, target_masks = self.extract_masked_features_batched(frame_tensors, mask_thresh=target_human_threshold, batch_size=batch_size)

        return self.compute_mean_distance(source_features, source_masks, target_features, target_masks, d)

    def compute_mean_distance(self, source_features, source_masks, target_features, target_masks, d='euclidean'):
        all_total_ds = []
        with tqdm(zip(source_features, source_masks, target_features, target_masks)) as frame_iter:
            for source, source_mask, target, target_mask in frame_iter:

                source_mean = np.mean(source[source_mask], axis=0)
                target_mean = np.mean(target[target_mask], axis=0)
                match d:
                    case 'euclidean':
                        d = np.linalg.norm(target_mean - source_mean)
                    case 'cosine':
                        d = - np.dot(target_mean, source_mean) / (np.linalg.norm(target_mean)*np.linalg.norm(source_mean))

                all_total_ds.append(d)

        return np.array(all_total_ds)

class Seq2SeqMetric:
    def __init__(self,
                map_fn,
                image2image_metric: Image2ImageMetric
                ):
        """
        map_fn computes a distance for each image in the source sequence, given a matrix representing distances between source and target
        image2image_metric computes a distance between two images
        """
        
        self.image2image_metric = image2image_metric
        self.seq2seq_fn = seq2seq_fn
  
    def forward(source_tensors, target_tensors, source_threshold=.001, target_threshold=.5, batch_size=16):

        source_features, source_masks = self.image2image_metric.extract_masked_features_batched(source_tensors, mask_thresh=source_threshold, batch_size=batch_size)
        target_features, target_masks = self.image2image_metric.extract_masked_features_batched(target_tensors, mask_thresh=target_threshold, batch_size=batch_size)
        
        source_indices, target_indices = torch.meshgrid([len(source_tensors), len(target_tensors)])

        all_source_features = source_features[source_indices]
        all_source_masks = source_masks[source_indices]

        all_target_features = target_features[target_indices]
        all_target_masks = target_masks[target_indices]

        all_ds = self.image2image_metric.compute_distance_parallel(all_source_features, all_source_masks, all_target_features, all_target_masks)
        all_ds = all_ds.view(len(source), len(target))

        source_costs = self.seq2seq_fn(all_ds)

        return source_costs 

    def forward_paths(source_path, target_path):
        source_gif_obj = Image.open(source_path)
        source_frames = load_gif_frames(source_gif_obj)

        target_gif_obj = Image.open(target_path)
        target_frames = load_gif_frames(target_gif_obj)

        ds = self.forward(source_frames, target_frames)

def compute_costs_per_source_dtw(distances):
    C, _ = compute_dtw(distances)
    
    # source_matches will contain each i in range(len(distances)) as a key
    source_matches = backtrack_dtw(C)

    source_costs = torch.zeros(len(source_matches))

    for source in source_matches.keys():
        targets = source_matches[source]

        # TODO: should it be an average over the matches? Or just the closest target frame to the source
        source_costs[source] = max(distances[source, targets]) 

    return source_costs

def compute_costs_per_source_wasserstein(distances):
    

    transport = ot.sinkhorn([], [], distances.cpu().numpy(), reg=1,log=False)
    return torch.as_tensor(transport).sum(dim=1)

def compute_dtw(distances) -> torch.Tensor:
    """
    DP approach to compute accumulated cost for dynamic time warp path 
    Returns the cost matrix and the total accumulated cost
    distances: a matrix containing each pairwise distance beween images in the two sequences
    """

    # Initialization
    cost = torch.zeros_like(distances)
    cost[0,0] = distances[0,0]
    
    for i in range(1, len(y)):
        cost[i, 0] = distances[i, 0] + cost[i-1, 0]  
        
    for j in range(1, len(x)):
        cost[0, j] = distances[0, j] + cost[0, j-1]  

    for i in range(1, len(y)):
        for j in range(1, len(x)):
            cost[i, j] = min(
                cost[i-1, j],    
                cost[i, j-1],    
                cost[i-1, j-1]   
            ) + distances[i, j] 
            
    return cost, cost[-1][-1]

def backtrack_dtw(cost):
    """
    Return optimal matches of each row to each col from the dtw cost matrix
    """
    i = len(cost)-1
    j = len(cost[0])-1

    matches = defaultdict(list)

    matches[i].append(j)

    default_cost = float("Inf")

    while i != 0 or j != 0:
        left = index_matrix_with_bounds(cost, i-1, j, default_cost)
        down = index_matrix_with_bounds(cost, i, j-1, default_cost)
        diag = index_matrix_with_bounds(cost, i-1, j-1, default_cost)

        if left < down and left < diag:
            matches[i-1].append(j)
        elif down < left and down < diag:
            matches[i].append(j-1)
        else:
            matches[i-1].append(j-1)

    return matches




def metric_factory(image_metric, 
                    feature_extractor,
                    patch_masker,
                    sequence_metric=None):
    """
    image_metric in ['mean_feature', 'wasserstein']
    sequence_metric in [None, 'wasserstein', 'dtw']
    if target_frame_path is listed, this will initialize the metric with the target frame already embedded
    """

    match image_metric:
        case 'mean_feature':
            image_metric_fn = PatchMeanFeatureDistance(feature_extractor=feature_extractor,
                                                        patch_masker=patch_masker)
        case 'wasserstein':
            image_metric_fn = PatchWassersteinDistance(feature_extractor=feature_extractor,
                                                        patch_masker=patch_masker)

    if sequence_metric is not None:
        match sequence_metric:
            case 'dtw':
                sequence_metric_fn = Seq2SeqMetric(compute_costs_per_source_dtw, image_metric_fn)
            case 'wasserstein':
                sequence_metric_fn = Seq2SeqMetric(compute_costs_per_source_dtw, image_metric_fn)
        return sequence_metric_fn

    return image_metric_fn


def gif_to_image_distances(gif_path, metric_fn, human_mask_thresh=.001, d_order=1):
    """
    Use metric_fn to calculate distances for each image in the gif specified by gif_path
    d_order is power to raise ds to after calculation (if you want to penalize outliers more)
    """
    source_gif = Image.open(gif_path)
    source_frames = load_gif_frames(source_gif)


    all_total_ds = metric_fn.forward_cached_target(source_frames, human_mask_thresh)
    if isinstance(metric_fn, SparseMatchingDistance) and metric_fn.plot==True:
        create_gif_from_figs(metric_fn.tmp_dir, 'outputs/matches.gif')                             

    return all_total_ds ** d_order



    

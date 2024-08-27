
import os
import numpy as np
import concurrent
from functools import lru_cache
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
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

# TODO (yuki): Have to change the path to work with the folder strucuture in CrossQ
#   Probably a better way to do this
from vlm_reward.utils.utils import index_matrix_with_bounds, create_gif_from_figs, timing_decorator

from loguru import logger

# import fastdtw

class Dino2FeatureExtractor:
    """
    Get patch level features from a DinoV2 model
    Crops input images to the correct size, and handles resizing projections later
    Adopted from: https://github.com/facebookresearch/dinov2/blob/255861375864acdd830f99fdae3d9db65623dafe/notebooks/features.ipynb
    """

    def __init__(self, 
                model_name="dinov2_vitb14", 
                repo_name="facebookresearch/dinov2", 
                edge_size=224,  # TODO (Yuki): To save some GPU memory, reduced from 448 to 224, could be the reason behind worse performance?
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

        def convert_from_uint8_to_float(image: torch.Tensor) -> torch.Tensor:
            if image.dtype == torch.uint8:
                return image.to(torch.float32) / 255.0
            else:
                return image
        
        self.transform = transforms.Compose([
            convert_from_uint8_to_float,
            transforms.Resize(size=edge_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(size=edge_size) # convert to square
        ])
                
        self.normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) # imagenet defaults

    def extract_features(self, images_tensor):
        with torch.no_grad():
            images_tensor = self.normalize(images_tensor)
            
            if len(images_tensor.shape) == 3:
                image_batch = images_tensor.unsqueeze(0)
            else:
                image_batch = images_tensor

            batch_size = images_tensor.size()[0]
            
            if self.half_precision:
                image_batch = image_batch.half()
            else:
                image_batch = image_batch

            image_batch = image_batch.to(self.device)
            all_tokens = self.model.get_intermediate_layers(image_batch)

            if batch_size != 1:
                tokens = all_tokens[0].squeeze()
            else:
                tokens = all_tokens[0]

        return tokens

    def extract_features_final(self, images_tensor):
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
            all_tokens = self.model.forward(image_batch)
        return all_tokens


    def prepare_images_parallel(self, images) -> torch.Tensor:
        """
        images: list of PIL.Images or Torch.Tensor
        Applies self.transform on the given batch of images using multithreading
        """
        # If a list of images or a tensor with 4 dimension is provided, 
        # use multi-threading for parallel processing
        if isinstance(images, list) or (isinstance(images, torch.Tensor) and len(images.shape) > 3):
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
            features, feature_level_human_mask = self.extract_masked_features(batch, use_patch_mask=mask_thresh ==0, mask_thresh=mask_thresh)
            all_features = torch.concat((all_features, features))
            all_masks = torch.concat((all_masks, feature_level_human_mask))

        return all_features, all_masks

    def extract_features_final(self, batch):
        return self.feature_extractor.extract_features_final(batch)

    def extract_masked_features(self, batch, use_patch_mask, mask_thresh=.5):
        """
        Extract masked features from a batch
        batch: (B, 3, H, W), where B is number of images in batch
        """
        # logger.debug(f"use_patch_mask={use_patch_mask}, mask_thresh={mask_thresh}")
        # Start human mask extraction first because interpolation is GPU-memory expensive
        if use_patch_mask:
            # logger.debug(f"{self.patch_masker.device=}, {batch.device=}")
            human_masks = self.patch_masker(batch, thresh=mask_thresh)[:,0,...]
        else:
            human_masks = torch.ones_like(batch)[:,0,...].bool()

        features = self.feature_extractor.extract_features(batch.to(self.device))

        _, n_features, _ = features.shape
        feature_indices = np.arange(n_features)
        grid_size = self.feature_extractor.get_grid_size(batch[0])
        rows, columns = self.feature_extractor.idx_to_source_position(feature_indices, grid_size, 1)
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
        return self.compute_distance_parallel(source_features, source_masks, target_features, target_masks, )

  
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

    def compute_distance_parallel(self, source_features, source_masks, target_features, target_masks, cost_fn='cosine', return_ot_plan=False):
        """
        Inputs are torch tensors
        """
        
        def compute_wasserstein_given_masks(source_feature, source_mask, target_feature, target_mask):
            """
            Inputs must be numpy arrays
            """
            return self.compute_patchwise_wasserstein(source_feature[source_mask],target_feature[target_mask], d=cost_fn)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = executor.map(compute_wasserstein_given_masks, 
                                    source_features.cpu().numpy(), 
                                    source_masks.cpu().numpy(),
                                    target_features.cpu().numpy(),
                                    target_masks.cpu().numpy()) 

            raw_output_dict = [future for future in futures]
            wassersteins = [o["wasser"] for o in raw_output_dict]
            if return_ot_plan:
                ot_plan = [o["T"] for o in raw_output_dict]
                costs = [o["C"] for o in raw_output_dict]

        if return_ot_plan:
            return dict(wasser=wassersteins, T=ot_plan, C=costs)
        else:
            return dict(wasser=wassersteins)

    
    @staticmethod
    def compute_patchwise_wasserstein(features, target_features, d='euclidean'):             
        """
        Return: a dictionary that contains field
            "wasser": the wasserstein distance
            "T": (optional) the optimal transport plan
        """   
        if len(features) == 0:
            print('Error: no source features found. Distance is considered -1. Consider decreasing the human threshold for the source or target')
            return dict(wasser = -1)
        elif len(target_features) == 0:
            print('Error: no target features found. Distance is considered -1. Consider decreasing the human threshold for the source or target')
            return dict(wasser = -1)

        match d:
            case 'cosine':
                similarity = np.dot(features, target_features.T) / np.linalg.norm(features, axis=1, keepdims=True) / np.linalg.norm(target_features.T, axis=0, keepdims=True) # Transpose B to match dimensions

                # from sklearn.metrics.pairwise import cosine_similarity
                # logger.debug(f"similarity=\n{similarity}\nsklearn_similarity=\n{cosine_similarity(features, target_features)}\nold=\n{M}")
                
                # Rescale to be between 0 and 1
                similarity_rescaled = (similarity + 1) / 2

                # Turn similarity into a cost
                M = 1 - similarity_rescaled

                # M = target_features @ features.T 
                # M *= 1 / np.tile(np.linalg.norm(target_features, axis=1), (M.shape[1], 1)).T
                # M *= 1 / np.tile(np.linalg.norm(features, axis=1), (M.shape[0], 1))
                # M *= -1 # turn the similarity into a cost

            case 'euclidean':
                M = cdist(target_features, features)
        
        # Uniform distribution
        features_weights = np.ones(features.shape[0]) * (1 / features.shape[0])
        target_features_weights = np.ones(target_features.shape[0]) * (1 / target_features.shape[0])
        
        # Sinkhorn2 directly outputs the distance (compared to sinkhorn which outputs the OT solution)
        # For regularizing, using this paper: (https://github.com/siddhanthaldar/ROT/blob/41ef7b98ca3950b9f31dd174f306cbe6916a09c9/ROT/rewarder.py#L4)
        T = ot.sinkhorn(features_weights, target_features_weights, M, reg=0.01, log=False)

        # Calculate the wasserstein distance (ref: )
        wasser = np.sum(T*M)

        return dict(wasser=wasser, T=T, C=M)

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

class FeatureDistance(Image2ImageMetric):

    def __init__(self,  
                feature_extractor, 
                patch_masker):

        super().__init__(feature_extractor, patch_masker)
    
    def compute_distance(self, source_features, target_features):
        all_total_ds = []
        return torch.linalg.vector_norm(target_features - source_features, dim=1).cpu().numpy()

class PatchMeanFeatureDistance(Image2ImageMetric):

    def __init__(self,  
                feature_extractor, 
                patch_masker):

        super().__init__(feature_extractor, patch_masker)

    def forward(self, source_frames, target_frames, source_human_threshold, target_human_threshold, d='euclidean'):
        source_features, source_masks = self.extract_masked_features(source_tensors, mask_thresh=source_threshold)
        target_features, target_masks = self.extract_masked_features(target_tensors, mask_thresh=target_threshold)
        return self.compute_mean_distance(source_features, source_masks, target_features, target_masks, d)

    def forward_batched(self, source_frames, target_frames, source_human_threshold, target_human_threshold, d='euclidean', batch_size=16):        
        source_features, source_masks = self.extract_masked_features_batched(frame_tensors, mask_thresh=source_human_threshold, batch_size=batch_size)
        target_features, target_masks = self.extract_masked_features_batched(frame_tensors, mask_thresh=target_human_threshold, batch_size=batch_size)

        return self.compute_mean_distance(source_features, source_masks, target_features, target_masks, d)

    def compute_distance_parallel(self, source_features, source_masks, target_features, target_masks, d='euclidean'):
        return self.compute_mean_distance(source_features.cpu().numpy(), 
                                        source_masks.cpu().numpy(), 
                                        target_features.cpu().numpy(),
                                        target_masks.cpu().numpy(), d)

    def compute_mean_distance(self, source_features, source_masks, target_features, target_masks, d_metric='euclidean'):
        all_total_ds = []
        with tqdm(zip(source_features, source_masks, target_features, target_masks)) as frame_iter:
            for source, source_mask, target, target_mask in frame_iter:

                source_mean = source[source_mask].mean(axis=0)
                target_mean = target[target_mask].mean(axis=0)
                match d_metric:
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
        self.seq2seq_fn = map_fn
        self.image2image_metric = image2image_metric
    
  
    def forward(self, source_images, target_images, source_threshold=.001, target_threshold=.5, batch_size=16):
        """
        images are PIL.Image.Image or torch.Tensor
        """ 
        source_tensors = self.image2image_metric.prepare_images(source_images)
        target_tensors = self.image2image_metric.prepare_images(target_images)

        # get the power set of source and target indices (to create cost matrix)
        source_indices, target_indices = torch.meshgrid([torch.arange(len(source_tensors)), torch.arange(len(target_tensors))])

        source_indices = source_indices.flatten()
        target_indices = target_indices.flatten()

        if isinstance(self.image2image_metric, FeatureDistance):
            source_features = self.image2image_metric.extract_features_final(source_tensors)
            target_features = self.image2image_metric.extract_features_final(target_tensors)
            all_ds = self.image2image_metric.compute_distance(source_features[source_indices], target_features[target_indices])
            
        else:
            source_features, source_masks = self.image2image_metric.extract_masked_features_batched(source_tensors, mask_thresh=source_threshold, batch_size=batch_size)
            target_features, target_masks = self.image2image_metric.extract_masked_features_batched(target_tensors, mask_thresh=target_threshold, batch_size=batch_size)

            all_source_features = source_features[source_indices]
            all_source_masks = source_masks[source_indices]
            all_target_features = target_features[target_indices]
            all_target_masks = target_masks[target_indices]
            all_ds = self.image2image_metric.compute_distance_parallel(all_source_features, all_source_masks, all_target_features, all_target_masks)
        
        all_ds = torch.as_tensor(all_ds)            
        all_ds = all_ds.view(len(source_tensors), len(target_tensors))

        source_costs = self.seq2seq_fn(all_ds)
        
       # normed_source_costs = F.normalize(source_costs, dim=0)

        return source_costs 

    def forward_paths(self, source_path, target_path):
        source_gif_obj = Image.open(source_path)
        source_frames = load_gif_frames(source_gif_obj)

        target_gif_obj = Image.open(target_path)
        target_frames = load_gif_frames(target_gif_obj)

        ds = self.forward(source_frames, target_frames)

def compute_costs_per_source_dtw(distances):
    """
    compute the cost per source image using dynamic time warping. Find the best target matches
    for each source image based on dtw, and then choose the target frame that is closest among those matches
    """
    C, _ = compute_dtw(distances)
    
    # source_matches will contain each i in range(len(distances)) as a key
    source_matches = backtrack_dtw(C)

    source_costs = torch.zeros(len(source_matches))

    for source in source_matches.keys():
        targets = source_matches[source]

        # TODO: should it be an average over the matches? Or just the closest target frame to the source
        source_costs[source] = min(distances[source, targets]) 

    return source_costs

def compute_costs_per_source_wasserstein(distances):
    """
    compute optimal transport cost for each source image, as described in RAPL
    i.e., sum the optimal transport matrix across the target dimension
    """
    distances_np = distances.cpu().numpy()


    transport = ot.sinkhorn([], [], distances_np, reg=1,log=False)

    # scale so that transport plan sums to 1 along target axis
    # ensures different source sequence lengths give same scaled outputs
    transport_scaled = transport * transport.shape[0] 

    target_weights = np.ones(distances_np.shape[1])
    costs = (transport_scaled * distances_np) @ target_weights
    return torch.as_tensor(costs)

def compute_dtw(distances) -> torch.Tensor:
    """
    DP approach to compute accumulated cost for dynamic time warp path 
    Returns the cost matrix and the total accumulated cost
    distances: a matrix containing each pairwise distance beween images in the two sequences
    """
    s, t = distances.shape

    # Initialization
    cost = torch.zeros_like(distances)
    cost[0,0] = distances[0,0]
    
    for i in range(1, s):
        cost[i, 0] = distances[i, 0] + cost[i-1, 0]  
        
    for j in range(1, t):
        cost[0, j] = distances[0, j] + cost[0, j-1]  

    for i in range(1, s):
        for j in range(1, t):
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
            i-=1
        elif down < left and down < diag:
            matches[i].append(j-1)
            j-=1
        else:
            matches[i-1].append(j-1)
            i-=1
            j-=1

    return matches




def metric_factory(image_metric, 
                    feature_extractor,
                    patch_masker,
                    sequence_metric=None):
    """
    image_metric in ['mean_patch_feature', 'wasserstein']
    sequence_metric in [None, 'wasserstein', 'dtw']
    if target_frame_path is listed, this will initialize the metric with the target frame already embedded
    """

    match image_metric:
        case 'mean_patch_feature':
            image_metric_fn = PatchMeanFeatureDistance(feature_extractor=feature_extractor,
                                                        patch_masker=patch_masker)
        case 'feature':   
            image_metric_fn = FeatureDistance(feature_extractor=feature_extractor,
                                                        patch_masker=patch_masker)
        case 'wasserstein':
            image_metric_fn = PatchWassersteinDistance(feature_extractor=feature_extractor,
                                                        patch_masker=patch_masker)
                                            

    if sequence_metric is not None:
        match sequence_metric:
            case 'dtw':
                sequence_metric_fn = Seq2SeqMetric(compute_costs_per_source_dtw, image_metric_fn)
            case 'wasserstein':
                sequence_metric_fn = Seq2SeqMetric(compute_costs_per_source_wasserstein, image_metric_fn)
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
                          
    return all_total_ds ** d_order



    

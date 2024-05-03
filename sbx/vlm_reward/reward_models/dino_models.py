
import os
import numpy as np
import concurrent
from functools import lru_cache


import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
import ot

import cv2
from PIL import Image
import imageio

from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

import matplotlib
from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import create_gif_from_figs, timing_decorator
from human_seg import HumanSegmentationModel


def load_dino_reward_model(dino_model_name, metric, human_seg_weight_path, target_human_threshold, tmp_dir):
    # TODO: We could load target frame into the model first
    target_frame = None

    match metric:
        case 'match':
            metric_fn = SparseMatchingDistance( 
                                            dino_model=dino_model_name,
                                            target_image=target_frame,
                                            target_human_threshold=target_human_threshold,
                                            human_seg_weight_path=human_seg_weight_path,
                                            plot=True,
                                            tmp_dir=tmp_dir)
        case 'mean_feature':
            metric_fn = MeanFeatureDistance(target_image=target_frame, 
                                            dino_model=dino_model_name,
                                            target_human_threshold=target_human_threshold,
                                            human_seg_weight_path=human_seg_weight_path)
        case 'wasserstein':
            metric_fn = PatchWassersteinDistance(dino_model=dino_model_name,
                                                human_seg_weight_path=human_seg_weight_path,
                                                target_image=target_frame, 
                                                target_human_threshold=target_human_threshold)

    return metric_fn


class Dino2FeatureExtractor:
    """
    Get patch level features from a DinoV2 model
    Crops input images to the correct size, and handles resizing projections later
    Adopted from: https://github.com/facebookresearch/dinov2/blob/255861375864acdd830f99fdae3d9db65623dafe/notebooks/features.ipynb
    """

    def __init__(self, repo_name="facebookresearch/dinov2", model_name="dinov2_vitb14", smaller_edge_size=448, half_precision=False, device="cuda"):
        self.repo_name = repo_name
        self.model_name = model_name
        self.smaller_edge_size = smaller_edge_size
        self.half_precision = half_precision
        self.device = device

        if self.half_precision:
            self.model = torch.hub.load(repo_or_dir=repo_name, model=model_name).half().to(self.device)
        else:
            self.model = torch.hub.load(repo_or_dir=repo_name, model=model_name).to(self.device)

        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(size=smaller_edge_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(size=smaller_edge_size), # convert to square
            transforms.ToTensor() 
        ])

        self.normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) # imagenet defaults

    def prepare_image(self, rgb_image_numpy):
        return self._prepare_single_image(rgb_image_numpy, self.transform, self.model.patch_size)

    @lru_cache(maxsize=3)
    def _load_and_prepare_single_image(self, image_path, transform, patch_size):
        image = Image.open(image_path).convert("RGB")
        return self._prepare_single_image(image, transform, patch_size)

    @staticmethod
    def _prepare_single_image(rgb_image_numpy, transform, patch_size):
        """
        Make static for parallel execution
        """
        if not isinstance(rgb_image_numpy, Image.Image):
            image = Image.fromarray(rgb_image_numpy)
        else:
            image = rgb_image_numpy
        image_tensor = transform(image)
        resize_scale = image.width / image_tensor.shape[2]

        # Crop image to dimensions that are a multiple of the patch size
        height, width = image_tensor.shape[1:] # C x H x W
        cropped_width, cropped_height = width - width % patch_size, height - height % patch_size # crop a bit from right and bottom parts
        image_tensor = image_tensor[:, :cropped_height, :cropped_width]

        grid_size = (cropped_height // patch_size, cropped_width // patch_size)
        return image_tensor, grid_size, resize_scale

    def prepare_images(self, images):
        processed_images = []
        for img in images:
            processed = self._prepare_single_image(img, self.transform, self.model.patch_size)[0]
            processed_images.append(processed)
        return torch.stack(processed_images).to(self.device)

    def load_and_prepare_images_parallel(self, paths):
        """
        paths: list of paths to images
        caches last three inputs, in case you call it on targets, then source, then targets again
        """
        transform = self.transform
        model_patch_size = self.model.patch_size
        

        prepare_fn = lambda path: self._load_and_prepare_single_image(path, transform, model_patch_size)

        # If a list of images is provided, use multi-threading for parallel processing
        if isinstance(paths, list):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                
                futures = executor.map(prepare_fn, paths)
                processed_images = [result[0] for result in futures]
            
            return torch.stack(processed_images)
        else:  # Single image case
            return self._load_and_prepare_single_image(paths, transform, model_patch_size)

    def prepare_images_parallel(self, images):
        """
        images: list of PIL.Images
        caches last three inputs, in case you call it on targets, then source, then targets again
        """
        transform = self.transform
        model_patch_size = self.model.patch_size
        prepare_fn = lambda img: self._prepare_single_image(img, transform, model_patch_size)

        # If a list of images is provided, use multi-threading for parallel processing
        if isinstance(images, list):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                
                futures = executor.map(prepare_fn, images)
                processed_images = [result[0] for result in futures]
            

            return torch.stack(processed_images)
        else:  # Single image case
            return self._prepare_single_image(images, transform, model_patch_size)


    def get_grid_size(self, img):
        C, H, W = img.shape

        return (H / self.model.patch_size, W / self.model.patch_size)

    def prepare_mask(self, mask_image_numpy, grid_size, resize_scale):
        cropped_mask_image_numpy = mask_image_numpy[:int(grid_size[0]*self.model.patch_size*resize_scale), :int(grid_size[1]*self.model.patch_size*resize_scale)]

        image = Image.fromarray(cropped_mask_image_numpy)
        resized_mask = image.resize((grid_size[1], grid_size[0]), resample=Image.Resampling.NEAREST)
        resized_mask = np.asarray(resized_mask).flatten()
        return resized_mask
    
    def extract_features(self, image_tensor):
        with torch.inference_mode():
            image_tensor = self.normalize(image_tensor)
            
            if len(image_tensor.shape) == 3:
                image_batch = image_tensor.unsqueeze(0)
            else:
                image_batch = image_tensor
            
            if self.half_precision:
                image_batch = image_batch.half()
            else:
                image_batch = image_batch

            image_batch = image_batch.to(self.device)
            
            all_tokens = self.model.get_intermediate_layers(image_batch)
            
            tokens = all_tokens[0].squeeze()
        return tokens
    

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


class DistanceMetric(nn.Module):
    def __init__(self, 
                dino_model, 
                human_seg_weight_path, 
                target_image=None, 
                target_human_threshold=.5,
                use_human_mask = True,
                device='cuda',
                **kwargs):
        """
        Compute distances between a target image and set of source images in dino feature space

        target_image: A PIL.Image, RGB
        dino_model: a model type specified in the dino repo (will download automatically)
        human_seg_weight_path: the path to the ResNet50 human segmentation weights
        target_human_threshold: the minimum pixel probability to consider it human (should be lower for different domains)
        """
        super().__init__(**kwargs)

        dm = Dino2FeatureExtractor(model_name=dino_model)
        human_seg_model = HumanSegmentationModel(human_seg_weight_path)

        self.dm = dm
        self.human_seg_model = human_seg_model
        self.use_human_mask=use_human_mask
        self.device = device

        if target_image is not None:
            self.update_target_image(target_image, target_human_threshold)

    def update_target_image(self, target_image, mask_thresh=.5):
        target_tensor, target_human_mask, target_features_masked, masked_target_feature_to_feature, target_grid_size, target_resize_scale = self.process_image(target_image, mask_thresh=mask_thresh)

        self.target_features_masked = target_features_masked
        self.target_human_mask = target_human_mask
        self.masked_target_features_to_feature = masked_target_feature_to_feature
        self.target_grid_size = target_grid_size
        self.target_resize_scale = target_resize_scale

    def process_image(self, image, mask_thresh):
        """
        Given an image and a human segmentation mask, obtain masked DINO features
        """
        tensor, grid_size, resize_scale = self.dm.prepare_image(image)
        features = self.dm.extract_features(tensor)

        if self.use_human_mask:
            human_mask = self.human_seg_model(tensor, thresh=mask_thresh)[0][0]
        else:
            human_mask = torch.ones_like(tensor)[0].bool()

        feature_level_human_mask = self.get_feature_level_human_mask(human_mask, features, grid_size)
        features_masked = features[feature_level_human_mask]
        masked_feature_to_feature = torch.nonzero(feature_level_human_mask)
        return tensor, human_mask, features_masked, masked_feature_to_feature, grid_size, resize_scale

    def extract_masked_features_batched(self, tensors, mask_thresh=.5, batch_size=16):
        """
        tensors: N_batches, 3, H, W
        caches the previous 3 results, in case you call extract(target), extract(source), and again extract(target)
        """
        if batch_size > 1:
            batch_tensors = torch.split(tensors, batch_size)
        else:
            batch_tensors = tensors

        all_features = torch.as_tensor([], device = self.device)
        all_masks = torch.as_tensor([], dtype=torch.bool, device=self.device)


        for tensor in batch_tensors:
            features = self.dm.extract_features(tensor)

            if batch_size > 1:
                _, n_features, _ = features.shape
            else:
                n_features, _ = features.shape

            feature_indices = np.arange(n_features)
            grid_size = self.dm.get_grid_size(tensors[0])
            rows, columns = self.dm.idx_to_source_position(feature_indices, grid_size, 1)

            if self.use_human_mask:
                human_masks = self.human_seg_model(tensor, thresh=mask_thresh)[:,0,...]
            else:
                human_masks = torch.ones_like(tensor)[:,0,...].bool()

            feature_level_human_mask = human_masks[:, rows, columns].bool()
            all_features = torch.concat((all_features, features))
            all_masks = torch.concat((all_masks, feature_level_human_mask))

        return all_features, all_masks


    def extract_features_from_sequence(self, sequence, mask_thresh):
        """
        Given a sequence of images, return just the masked dino features for each image in the sequence
        """
        features = []
        for frame in sequence:
            _, _, features_masked, _, _, _ = self.process_image(frame, mask_thresh=mask_thresh)
            features.append(features_masked)
        return features

    def get_feature_level_human_mask(self, human_mask, features, grid_size):
        feature_level_human_mask = torch.zeros(len(features))
        for idx in range(len(features)):
            r, c = self.dm.idx_to_source_position(idx, grid_size, 1)
            feature_level_human_mask[idx] = human_mask[int(r), int(c)]
        return feature_level_human_mask.bool()

class PatchWassersteinDistance(DistanceMetric):

    def __init__(self, 
                dino_model, 
                human_seg_weight_path,
                target_image=None, 
                target_human_threshold=.5):

        super().__init__(dino_model, human_seg_weight_path, target_image, target_human_threshold)
  
    @timing_decorator
    def forward(self, frames, human_threshold, d='euclidean', batch_size=16):
        """
        Obtain wasserstein distances between all frames and the current target frame
        Allows for caching of the target frame features
        human_threshold is used to mask the input frames
        """
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        frame_tensors = self.dm.prepare_images_parallel(frames).to(self.device)
        end_event.record()
        
        ### timing
        torch.cuda.synchronize()
        execution_time = start_event.elapsed_time(end_event)
        print(f"Image loading time: {execution_time / 1000} seconds")
        ####

        start_event.record()
        all_features, all_masks = self.extract_masked_features_batched(frame_tensors, human_threshold, batch_size=batch_size)
        end_event.record()

        ### timing
        torch.cuda.synchronize()
        execution_time = start_event.elapsed_time(end_event)
        print(f"Feature extraction time: {execution_time / 1000} seconds")
        ###
        
        start_event.record()

        # wassersteins = []
        # for feature,mask in zip(all_features, all_masks):
        #     wassersteins.append(self.compute_patchwise_wasserstein(self.target_features_masked.cpu().numpy(), feature[mask].cpu().numpy()))

        target_features_masked_np = self.target_features_masked.cpu().numpy()

        def compute_wasserstein_source(feature, mask):
            return self.compute_patchwise_wasserstein(target_features_masked_np, feature[mask], d=d)


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

    def forward_paths(self, source_paths, target_paths, source_threshold=.001, target_threshold=.5, d='euclidean', batch_size=16):
        source_tensors = self.dm.load_and_prepare_images_parallel(source_paths).to(self.device)
        target_tensors = self.dm.load_and_prepare_images_parallel(target_paths).to(self.device)
        return self._forward_batched_tensors(source_tensors, target_tensors, source_threshold, target_threshold, d, batch_size)

    def _forward_batched_tensors(self, source_tensors, target_tensors, source_threshold, target_threshold, d, batch_size):
        """
        Compute wasserstein distance between a batch of source frames and target frames
        Returns [d(source_frames[i], target_frames[i]) for all i]
        source_threshold and target_threshold are used for the human masking model
        """

        source_features, source_masks = self.extract_masked_features_batched(source_tensors, source_threshold, batch_size=batch_size)
        target_features, target_masks = self.extract_masked_features_batched(target_tensors, target_threshold, batch_size=batch_size)

        def compute_wasserstein_given_masks(source_feature, source_mask, target_feature, target_mask):
            """
            Inputs must be numpy arrays
            """
            return self.compute_patchwise_wasserstein(target_feature[target_mask], source_feature[source_mask], d=d)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = executor.map(compute_wasserstein_given_masks, 
                                    source_features.cpu().numpy(), 
                                    source_masks.cpu().numpy(),
                                    target_features.cpu().numpy(),
                                    target_masks.cpu().numpy()) 
            wassersteins = [future for future in futures]
        

        return wassersteins

    @staticmethod
    def compute_patchwise_wasserstein(target_features, features, d='euclidean'):
                        
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
            return 0
        else:
            # for now, use constant weights
            target_weights = []
            source_weights = []

            ### GROMOV WASSERSTEIN
            # C1 = cdist(features, features)
            # C2 = cdist(target_features, target_features)
            

            # coupling = ot.gromov.entropic_gromov_wasserstein(C1, C2)
            # breakpoint()
            # wasser = coupling @ C2 @ C1

            
            # for some reason, ot.sinkhorn2 is much faster on cpu. There must be a better way than this
            wasser= ot.sinkhorn2(target_weights, source_weights, M, reg=1,log=False)
            
        return wasser


    # TODO: define an embed_module
    def embed_module(self, frames):
        """Assume the frames here are a batch
        """
        # Transform the frames in batch
        

        # Call DINOv2 to encode the image

    # TODO: Define a __call__ fn so that it's easier to call by the SAc

   
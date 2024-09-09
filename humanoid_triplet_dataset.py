import os
import json
from PIL import Image
import datasets
import numpy as np

DATA_DIR = '/share/portal/aw588/finetuning/data'  # Root directory of your dataset

class HumanoidTripletDataset(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            description="Triplet dataset containing anchor, positive, and negative images",
            features=datasets.Features({
                'anchor_image': datasets.Image(),  # Storing the anchor image
                'anchor_geom_xpos': datasets.Sequence(
                datasets.Sequence(datasets.Value("float32"), length=3), length=18), # Storing the anchor joint's global positions (18,3)
                'pos_image': datasets.Image(),     # Storing the positive image
                'pos_geom_xpos': datasets.Sequence(
                datasets.Sequence(datasets.Value("float32"), length=3), length=18), # Storing the positive joint's global positions (18,3)
                'neg_image': datasets.Image(),     # Storing the negative image
                'neg_geom_xpos': datasets.Sequence(
                datasets.Sequence(datasets.Value("float32"), length=3), length=18) # Storing the negative joint's global positions (18,3)
            }),
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={
                'filepath': os.path.join(DATA_DIR, 'train_split.json'),
            }),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={
                'filepath': os.path.join(DATA_DIR, 'val_split.json'),
            }),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={
                'filepath': os.path.join(DATA_DIR, 'test_split.json'),
            }),
            datasets.SplitGenerator(name='manual_test', gen_kwargs={
                'filepath': os.path.join(DATA_DIR, 'manual_test_split.json'),
            }),
        ]
    
    def _convert_image_path_to_geom_xpos_path(self, geom_xpos_path):
        """
        Anchor image
            v3_body_distortion_arm, v3_flipping, v3_random_joints
                Image has the format {id}_pose.png, but it's geom_xpos has the format {id}_geom_xpos.npy
            v3_seq
                Image has the format {seqence_name}.png, but it's geom_xpos has the format {sequence_name}_geom_xpos.npy
        Positive image
            v3_body_distortion_arm, v3_flipping, v3_random_joints, v3_seq
                Image has the format {image_name}.png, but it's geom_xpos has the format {image_name}_geom_xpos.npy
        Negative image
            v3_body_distortion_arm, v3_flipping, v3_random_joints, v3_seq
                Image has the format {image_name}.png, but it's geom_xpos has the format {image_name}_geom_xpos.npy
        """
        if "anchor" in geom_xpos_path and ("v3_body_distortion_arm" in geom_xpos_path or "v3_flipping" in geom_xpos_path or "v3_random_joints" in geom_xpos_path):
            return geom_xpos_path.replace("_pose.png", "_geom_xpos.npy")
        else:
            return geom_xpos_path.replace(".png", "_geom_xpos.npy")


    def _generate_examples(self, filepath):
        # Open the JSON file for the given split and load the image paths
        with open(filepath, 'r') as f:
            data = json.load(f)  # Load JSON data into a Python list of dictionaries
            
            for id_, item in enumerate(data):
                # Extract the file paths for the anchor, positive, and negative images
                if DATA_DIR not in item['anchor']:
                    anchor_image_path = item['anchor'].replace('finetuning/data', DATA_DIR)
                else:
                    anchor_image_path = item['anchor']
                if DATA_DIR not in item['pos']:
                    pos_image_path = item['pos'].replace('finetuning/data', DATA_DIR)
                else:
                    pos_image_path = item['pos']
                if DATA_DIR not in item['neg']:
                    neg_image_path = item['neg'].replace('finetuning/data', DATA_DIR)
                else:
                    neg_image_path = item['neg']

                # Load images using PIL
                anchor_image = Image.open(anchor_image_path).convert("RGB")
                pos_image = Image.open(pos_image_path).convert("RGB")
                neg_image = Image.open(neg_image_path).convert("RGB")

                # Load the global positions of the joints
                anchor_geom_xpos_path = np.load(self._convert_image_path_to_geom_xpos_path(anchor_image_path))
                pos_geom_xpos_path = np.load(self._convert_image_path_to_geom_xpos_path(pos_image_path))
                neg_geom_xpos_path = np.load(self._convert_image_path_to_geom_xpos_path(neg_image_path))
                
                yield id_, {
                    'anchor_image': anchor_image,
                    'anchor_geom_xpos': anchor_geom_xpos_path,
                    'pos_image': pos_image,
                    'pos_geom_xpos': pos_geom_xpos_path,
                    'neg_image': neg_image,
                    'neg_geom_xpos': neg_geom_xpos_path
                }
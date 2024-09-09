import os
import json
from PIL import Image
import datasets

# You can either provide the full paths here or use dl_manager to handle downloading if applicable
DATA_DIR = '/share/portal/aw588/finetuning/data'  # Root directory of your dataset


class HumanoidTripletDataset(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            description="Triplet dataset containing anchor, positive, and negative images",
            features=datasets.Features({
                'anchor_image': datasets.Image(),  # Storing the anchor image
                'pos_image': datasets.Image(),     # Storing the positive image
                'neg_image': datasets.Image(),     # Storing the negative image
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

    def _generate_examples(self, filepath):
        # Open the JSON file for the given split and load the image paths
        with open(filepath, 'r') as f:
            data = json.load(f)  # Load JSON data into a Python list of dictionaries
            
            for id_, item in enumerate(data):
                # Extract the file paths for the anchor, positive, and negative images
                anchor_image_path = item['anchor'].replace('finetuning/data', DATA_DIR)
                pos_image_path = item['pos'].replace('finetuning/data', DATA_DIR)
                neg_image_path = item['neg'].replace('finetuning/data', DATA_DIR)

                # Load images using PIL
                anchor_image = Image.open(anchor_image_path).convert("RGB")
                pos_image = Image.open(pos_image_path).convert("RGB")
                neg_image = Image.open(neg_image_path).convert("RGB")
                
                yield id_, {
                    'anchor_image': anchor_image,
                    'pos_image': pos_image,
                    'neg_image': neg_image,
                }
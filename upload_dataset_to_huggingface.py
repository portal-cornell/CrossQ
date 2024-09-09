import os
import json
from PIL import Image
import datasets

from datasets import load_dataset, DatasetDict, Dataset
import humanoid_triplet_dataset  # Import your dataset script

from tqdm import tqdm

import numpy as np

if __name__ == "__main__":
    # Path to your local dataset directory
    data_dir = "/share/portal/aw588/finetuning/data"  # This is where your JSON files and image files are stored

    # Load the dataset using the custom dataset script
    # Remove streaming for now so that we can check what we got
    dataset = load_dataset('humanoid_triplet_dataset.py', split="manual_test", data_dir=data_dir, trust_remote_code=True)

    # Access and display an example
    example = dataset[0]
    anchor_image = example['anchor_image']
    pos_image = example['pos_image']
    neg_image = example['neg_image']

    anchor_image.save('anchor_image.png')
    pos_image.save('pos_image.png')
    neg_image.save('neg_image.png')

    # Warning: when you load the geom_xpos, it will be a list of lists of lists
    #   Need to convert it to a numpy array
    print(f"anchor image geom xpos shape: {np.array(example['anchor_geom_xpos']).shape}\n{example['anchor_geom_xpos']}")
    print(f"pos image geom xpos shape: {np.array(example['pos_geom_xpos']).shape}\n{example['pos_geom_xpos']}")
    print(f"neg image geom xpos shape: {np.array(example['neg_geom_xpos']).shape}\n{example['neg_geom_xpos']}")

    # dataset.push_to_hub("LunaY0Yuki/humanoid_triplet_dataset")

    # Function to convert IterableDataset to a regular Dataset
    # def convert_iterable_to_dataset(iterable_dataset):
    #     # Create an empty dictionary to hold lists of column data
    #     examples = {
    #         'anchor_image': [],
    #         'pos_image': [],
    #         'neg_image': []
    #     }
        
    #     # Iterate through the IterableDataset and collect data into the lists
    #     for example in tqdm(iterable_dataset):
    #         examples['anchor_image'].append(example['anchor_image'])
    #         examples['pos_image'].append(example['pos_image'])
    #         examples['neg_image'].append(example['neg_image'])
        
    #     # Convert the list of examples into a Dataset
    #     return Dataset.from_dict(examples)

    # # Check if the dataset is IterableDataset and convert it
    # train_dataset = dataset["train"]
    # if isinstance(train_dataset, datasets.IterableDataset):
    #     print("Converting train dataset to regular Dataset")
    #     train_dataset = convert_iterable_to_dataset(train_dataset)

    # val_dataset = dataset["validation"]
    # if isinstance(val_dataset, datasets.IterableDataset):
    #     print("Converting validation dataset to regular Dataset")
    #     val_dataset = convert_iterable_to_dataset(val_dataset)

    # test_dataset = dataset["test"]
    # if isinstance(test_dataset, datasets.IterableDataset):
    #     print("Converting test dataset to regular Dataset")
    #     test_dataset = convert_iterable_to_dataset(test_dataset)

    # manual_test_dataset = dataset["manual_test"]
    # if isinstance(manual_test_dataset, datasets.IterableDataset):
    #     manual_test_dataset = convert_iterable_to_dataset(manual_test_dataset)

    # # Create a DatasetDict after conversion
    # dataset_dict = DatasetDict({
    #     "train": train_dataset,
    #     "validation": val_dataset,
    #     "test": test_dataset,
    #     "manual_test": manual_test_dataset
    # })
    
    # # Push the dataset to Hugging Face Hub
    # dataset_dict.push_to_hub("LunaY0Yuki/humanoid_triplet_dataset")

    
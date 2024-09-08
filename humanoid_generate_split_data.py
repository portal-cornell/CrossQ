"""
Split the generated data into train/val/test splits, while carefully avoid
overlap with the existing manual test set.
"""

import os
import os.path as op
import json
import random
from tqdm import tqdm
from loguru import logger

from utils_data_gen.utils_humanoid_generate import set_seed

set_seed(1231)


IMAGE_TYPE = "v3_seq"

INPUT_PATH = f"finetuning/data/{IMAGE_TYPE}"
ANCHOR_PATH = f"finetuning/data/{IMAGE_TYPE}/anchor"
NEG_PATH = f"finetuning/data/{IMAGE_TYPE}/neg"
POS_PATH = f"finetuning/data/{IMAGE_TYPE}/pos"
MANUAL_TEST_PATH = f"finetuning/data/{IMAGE_TYPE}/debug_50_manually_checked/"

OUTPUT_PATH = INPUT_PATH

# Get the list of all images in the manual test set
manual_test_images = [f for f in os.listdir(MANUAL_TEST_PATH) if f.endswith(".png")]
manual_prefix = [name.split("_eval.png")[0] for name in manual_test_images]
manual_prefix_to_data = {
    name.split("_eval.png")[0]: op.join(ANCHOR_PATH, [f for f in os.listdir(ANCHOR_PATH) if f.startswith(name.split("_eval.png")[0]) and f.endswith(".png")][0])
    for name in manual_test_images
}
print(f"Found {len(manual_prefix)} images in the manual test set.")

# Get the list of all anchor images
anchor_images = [f for f in os.listdir(ANCHOR_PATH) if f.endswith(".png")]
anchor_prefix = [name.split("_anchor")[0] for name in anchor_images]
anchor_prefix_to_data = {
    name.split("_anchor")[0]: op.join(ANCHOR_PATH, name)
    for name in anchor_images
}
print(f"Found {len(anchor_prefix)} anchor images.")

# Filter out the anchor images from the manual test set
filtered_anchor_prefix = [prefix for prefix in anchor_prefix if prefix not in manual_prefix]
print(f"Filtered {len(anchor_prefix) - len(filtered_anchor_prefix)} anchor images.")

# Sample 5k for train, 2k for val, 2k for test
random.shuffle(filtered_anchor_prefix)

train_prefix = filtered_anchor_prefix[:5000]
val_prefix = filtered_anchor_prefix[5000:7000]
test_prefix = filtered_anchor_prefix[7000:]

def create_split(prefix_list, path_type, prefix_d):
    output = []
    for prefix in tqdm(prefix_list):
        anchor_path = prefix_d[prefix]
        # import pdb; pdb.set_trace()
        pos_path = op.join(POS_PATH, [f for f in os.listdir(POS_PATH) if f.startswith(prefix) and f.endswith(".png")][0])
        neg_path = op.join(NEG_PATH, [f for f in os.listdir(NEG_PATH) if f.startswith(prefix) and f.endswith(".png")][0])
        
        if op.exists(anchor_path) and op.exists(pos_path) and op.exists(neg_path):
            output.append({
                "anchor": anchor_path,
                "pos": pos_path,
                "neg": neg_path
            })
        else:
            if not op.exists(anchor_path):
                raise ValueError(f"Missing anchor image for prefix {prefix}")
            if not op.exists(pos_path):
                raise ValueError(f"Missing pos image for prefix {prefix}")
            if not op.exists(neg_path):
                raise ValueError(f"Missing neg image for prefix {prefix}")

    return output

# Create splits
logger.info("Creating train split")
train_split = create_split(train_prefix, "train", anchor_prefix_to_data)
logger.info("Creating val split")
val_split = create_split(val_prefix, "val", anchor_prefix_to_data)
logger.info("Creating test split")
test_split = create_split(test_prefix, "test", anchor_prefix_to_data)
logger.info("Creating manual test split")
manual_test_split = create_split(manual_prefix, "manual_test", manual_prefix_to_data)


# Print split sizes
logger.info(f"Train split size: {len(train_split)}")
logger.info(f"Val split size: {len(val_split)}")
logger.info(f"Test split size: {len(test_split)}")
logger.info(f"Manual test split size: {len(manual_test_split)}")

# Save splits to JSON files
def save_split(split_data, split_name):
    output_path = op.join(INPUT_PATH, f"{split_name}_split.json")
    if op.exists(output_path):
        import pdb
        pdb.set_trace()
    else:
        with open(output_path, 'w') as f:
            json.dump(split_data, f, indent=2)
        print(f"Saved {split_name} split to {output_path}")

save_split(train_split, "train")
save_split(val_split, "val")
save_split(test_split, "test")
save_split(manual_test_split, "manual_test")

logger.info("Done")

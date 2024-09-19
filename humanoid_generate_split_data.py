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


# IMAGE_TYPE = "v3_random_joints" # v3_flipping, v3_seq, v3_random_joints
# IMAGE_TYPE = "v4_seq_frame25-3070"
# IMAGE_TYPE = "v4_seq_frame20-30_40-55"
IMAGE_TYPE = "v4_seq_frame20-30_40-60_all"

INPUT_PATH = f"finetuning/data/{IMAGE_TYPE}"
ANCHOR_PATH = f"finetuning/data/{IMAGE_TYPE}/anchor"
NEG_PATH = f"finetuning/data/{IMAGE_TYPE}/neg"
POS_PATH = f"finetuning/data/{IMAGE_TYPE}/pos"
if IMAGE_TYPE == "v3_flipping":
    # We'll add the 4 with target images
    MANUAL_TEST_PATH = f"finetuning/data/{IMAGE_TYPE}/debug_46_manually_checked/"
else:
    MANUAL_TEST_PATH = f"finetuning/data/{IMAGE_TYPE}/debug_50_manually_checked/"

OUTPUT_PATH = INPUT_PATH

# # Get the list of all images in the manual test set
# manual_test_images = [f for f in os.listdir(MANUAL_TEST_PATH) if f.endswith(".png")]
# if IMAGE_TYPE == "v3_flipping":
#     manual_prefix = [name.split("flipping_triplet_")[1].split(".png")[0] for name in manual_test_images]
#     manual_prefix_to_data = {
#         name.split("flipping_triplet_")[1].split(".png")[0]: op.join(ANCHOR_PATH, [f for f in os.listdir(ANCHOR_PATH) if f.startswith(name.split("flipping_triplet_")[1].split(".png")[0]) and f.endswith(".png")][0])
#         for name in manual_test_images
#     }
# elif "_seq" in IMAGE_TYPE:
#     manual_prefix = [name.split("_eval.png")[0] for name in manual_test_images]
#     manual_prefix_to_data = {
#         name.split("_eval.png")[0]: op.join(ANCHOR_PATH, [f for f in os.listdir(ANCHOR_PATH) if f.startswith(name.split("_eval.png")[0]) and f.endswith(".png")][0])
#         for name in manual_test_images
#     }
# else:
#     manual_prefix = [name.split("samples_")[1].split(".png")[0] for name in manual_test_images]
#     manual_prefix_to_data = {
#         name.split("samples_")[1].split(".png")[0]: op.join(ANCHOR_PATH, [f for f in os.listdir(ANCHOR_PATH) if f.startswith(name.split("samples_")[1].split(".png")[0]) and f.endswith(".png")][0])
#         for name in manual_test_images
#     }
# print(f"Found {len(manual_prefix)} images in the manual test set.")

# Get the list of all anchor images
anchor_images = [f for f in os.listdir(ANCHOR_PATH) if f.endswith(".png")]
anchor_prefix = [name.split("_anchor")[0] for name in anchor_images]
anchor_prefix_to_data = {
    name.split("_anchor")[0]: op.join(ANCHOR_PATH, name)
    for name in anchor_images
}
print(f"Found {len(anchor_prefix)} anchor images.")

# # Filter out the anchor images from the manual test set
filtered_anchor_prefix = [prefix for prefix in anchor_prefix] # if prefix not in manual_prefix]
# print(f"Filtered {len(anchor_prefix) - len(filtered_anchor_prefix)} anchor images.")

# Sample 5k for train, 2k for val, 2k for test
random.shuffle(filtered_anchor_prefix)

total_num = len(filtered_anchor_prefix)

train_prefix = filtered_anchor_prefix[:int(total_num * 0.9)]
val_prefix = filtered_anchor_prefix[int(total_num * 0.9):int(total_num * 0.95)]
test_prefix = filtered_anchor_prefix[int(total_num * 0.95):]
# test_prefix = anchor_prefix

def create_split(prefix_list, path_type, prefix_d):
    output = []
    for prefix in tqdm(prefix_list):
        anchor_path = prefix_d[prefix]
        if "_seq" in IMAGE_TYPE:
            pos_path = op.join(POS_PATH, [f for f in os.listdir(POS_PATH) if f.startswith(prefix) and f.endswith(".png")][0])
            neg_path = op.join(NEG_PATH, [f for f in os.listdir(NEG_PATH) if f.startswith(prefix) and f.endswith(".png")][0])
        elif IMAGE_TYPE in ["v3_flipping", "v3_random_joints"]:
            pos_path = op.join(POS_PATH, [f for f in os.listdir(POS_PATH) if f.startswith(prefix.split("_pose.png")[0]) and f.endswith(".png")][0])
            neg_path = op.join(NEG_PATH, [f for f in os.listdir(NEG_PATH) if f.startswith(prefix.split("_pose.png")[0]) and f.endswith(".png")][0])

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
    
    # For flipping we need to include the triplets with the target images
    # Path: data/v3_flipping/manual_test/anchor
    # The images for anchor/pos/neg have exactly the same name (so no need of prefix), just that they're in different folders
    if IMAGE_TYPE == "v3_flipping" and path_type == "manual_test":
        flipping_output = []
        flipping_base_path = "finetuning/data/v3_flipping/manual_test"
        # TODO: to normalize for easier parsing
        for image_name in [f for f in os.listdir(op.join(flipping_base_path, "anchor")) if f.endswith(".png")]:
            image_basename = image_name.split("0_pose_")[1]
            anchor_path = op.join(flipping_base_path, "anchor", image_name)
            pos_path = op.join(flipping_base_path, "pos", [f for f in os.listdir(op.join(flipping_base_path, "pos")) if image_basename in f and f.endswith(".png")][0])
            neg_path = op.join(flipping_base_path, "neg", [f for f in os.listdir(op.join(flipping_base_path, "neg")) if image_basename in f and f.endswith(".png")][0])
            
            if op.exists(anchor_path) and op.exists(pos_path) and op.exists(neg_path):
                flipping_output.append({
                    "anchor": anchor_path,
                    "pos": pos_path,
                    "neg": neg_path
                })
            else:
                if not op.exists(anchor_path):
                    raise ValueError(f"Missing anchor image for prefix {prefix}, {anchor_path}")
                if not op.exists(pos_path):
                    raise ValueError(f"Missing pos image for prefix {prefix}, {pos_path}")
                if not op.exists(neg_path):
                    raise ValueError(f"Missing neg image for prefix {prefix}, {neg_path}")
                logger.warning(f"Missing image(s) for {image_name} in v3_flipping manual test set")
        
        output.extend(flipping_output)

    return output

# Create splits
logger.info("Creating train split")
train_split = create_split(train_prefix, "train", anchor_prefix_to_data)
logger.info("Creating val split")
val_split = create_split(val_prefix, "val", anchor_prefix_to_data)
logger.info("Creating test split")
test_split = create_split(test_prefix, "test", anchor_prefix_to_data)
# logger.info("Creating manual test split")
# manual_test_split = create_split(manual_prefix, "manual_test", manual_prefix_to_data)


# Print split sizes
logger.info(f"Train split size: {len(train_split)}")
logger.info(f"Val split size: {len(val_split)}")
logger.info(f"Test split size: {len(test_split)}")
# logger.info(f"Manual test split size: {len(manual_test_split)}")

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
# save_split(manual_test_split, "manual_test")

logger.info("Done")

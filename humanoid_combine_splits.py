import json
import os
from pathlib import Path
from loguru import logger

IMAGE_TYPES = ["v3_flipping", "v3_seq", "v3_random_joints", "v3_body_distortion_arm"]
SPLIT_TYPES = ["train", "val", "test", "manual_test"]

BASE_PATH = "/share/portal/aw588/finetuning/data"
# OUTPUT_PATH = Path(BASE_PATH) / "combined_splits"
# OUTPUT_PATH.mkdir(exist_ok=True)
OUTPUT_PATH = Path(BASE_PATH)

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

combined_splits = {split_type: [] for split_type in SPLIT_TYPES}

for image_type in IMAGE_TYPES:
    logger.info(f"Processing {image_type}")
    for split_type in SPLIT_TYPES:
        input_file = Path(BASE_PATH) / image_type / f"{split_type}_split.json"
        if input_file.exists():
            data = load_json(input_file)
            combined_splits[split_type].extend(data)
            logger.info(f"  Added {len(data)} items to {split_type} split")
        else:
            logger.warning(f"  {input_file} not found. Skipping.")

for split_type in SPLIT_TYPES:
    output_file = OUTPUT_PATH / f"{split_type}_split.json"
    save_json(combined_splits[split_type], output_file)
    logger.info(f"Saved combined {split_type} split with {len(combined_splits[split_type])} items to {output_file}")

logger.info("Combination complete!")
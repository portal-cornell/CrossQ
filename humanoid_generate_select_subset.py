# from utils_data_gen.utils_humanoid_generate import select_random_debug_samples

import shutil, os, random

def select_random_debug_samples(source_folder, dest_folder, num_samples=200):
    os.makedirs(dest_folder, exist_ok=True)

    all_files = [f for f in os.listdir(source_folder) if f.endswith('.png')]
    selected_files = random.sample(all_files, min(num_samples, len(all_files)))

    for file in selected_files:
        shutil.copy(os.path.join(source_folder, file), os.path.join(dest_folder, file))

    print(f"Copied {len(selected_files)} files to {dest_folder}")


if __name__ == "__main__":

    IMAGE_TYPE = "v3_random_joints"

    if IMAGE_TYPE == "v3_seq":
        OUTPUT_ROOT = "finetuning/data/"
        FOLDER = f"{OUTPUT_ROOT}/v3_seq"
        select_random_debug_samples(f"{FOLDER}/debug", f"{FOLDER}/debug_50", num_samples=50)
    
    elif IMAGE_TYPE == "v3_flipping":
        OUTPUT_ROOT = "finetuning/data/"
        FOLDER = f"{OUTPUT_ROOT}/v3_flipping"
        # Only 46 kept because we include the 4 from target images
        select_random_debug_samples(f"{FOLDER}/debug", f"{FOLDER}/debug_75", num_samples=75)

    elif IMAGE_TYPE == "v3_random_joints":
        OUTPUT_ROOT = "finetuning/data/"
        FOLDER = f"{OUTPUT_ROOT}/v3_random_joints"
        select_random_debug_samples(f"{FOLDER}/debug", f"{FOLDER}/debug_75", num_samples=75)
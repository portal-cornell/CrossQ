import os

# Delete folder that (1) doesn't have any .gif and .npy files or (2) has a checkpoint with more than 3 models

def clean_folder(folder_path: str) -> None:
    """
    Parameters:
        folder_path: str
            - The path to the folder that holds all the logs/outputs for the current run

    Returns:
        bool
            - True if the folder has any .gif and .npy files
            - False if the folder doesn't have any .gif and .npy files
    """
    if "before_migration_to_hydra" in folder_path:
        return False

    # Check if the folder has any .gif and .npy files
    if os.path.exists(os.path.join(folder_path, "eval")):
        # Get all the files in the folder
        files = os.listdir(os.path.join(folder_path, "eval"))
        # Check if there are any .gif and .npy files
        files = [file for file in files if ".gif" in file or ".npy" in file]
        if len(files) != 0:
            return True

    # There are some earlier folders that don't have the eval folder
    if os.path.exists(os.path.join(folder_path, "checkpoint")):
        # Get all the files in the folder
        files = os.listdir(os.path.join(folder_path, "checkpoint"))
        # Check if there is a checkpoint with more than 3 models
        models = [file for file in files if ".zip" in file]
        if len(models) >= 3:
            return False
        
    # If the folder doesn't meet the conditions, delete it
    print("\nAbout to delete this folder: ", folder_path)
    print(">>>>> Files in the folder: ", os.listdir(folder_path))
    if os.path.exists(os.path.join(folder_path, "eval")):
        print("In eval: ", os.listdir(os.path.join(folder_path, "eval")))
    if os.path.exists(os.path.join(folder_path, "checkpoint")):
        print("In checkpoint: ", os.listdir(os.path.join(folder_path, "checkpoint")))
    input("Press any key to continue...")
    os.system(f"rm -rf {folder_path}")

    return False


def get_directory_size(directory):
    """Calculate the total size of a directory in bytes."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # Check if file exists to avoid errors
            if os.path.exists(fp):
                total_size += os.path.getsize(fp)
    return total_size

def show_top_level_directory_sizes(base_dir):
    # Get a list of all top-level directories in the base directory
    folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]

    # Loop through the folders and calculate their size
    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        folder_size = get_directory_size(folder_path)
        print(f"Folder: {folder}, Size: {folder_size / (1024 * 1024):.2f} MB")



if __name__ == "__main__":
    folder_path = "/share/portal/wph52/CrossQ"
    show_top_level_directory_sizes(folder_path)
    

    has_gif_folder_list = []

    # for f in os.listdir(folder_path):
    #     has_gif = clean_folder(os.path.join(folder_path, f))
        
    #     if has_gif:
    #         has_gif_folder_list.append(os.path.join(folder_path, f, "eval"))

    # print(has_gif_folder_list)


    # print("==============")
    # print([f for f in has_gif_folder_list if "both_arms_out" in f or "left_arm_out" in f or "right_arm_out" in f or "left_arm_extend_wave_higher" in f or "right_arm_extend_wave_higher" in f])
    # print()
    # print(len([f for f in has_gif_folder_list if "both_arms_out" in f or "left_arm_out" in f or "right_arm_out" in f or "left_arm_extend_wave_higher" in f or "right_arm_extend_wave_higher" in f]))

    # print("==============")
    # print([f for f in has_gif_folder_list if ("both_arms_out" in f or "left_arm_out" in f or "right_arm_out" in f or "left_arm_extend_wave_higher" in f or "right_arm_extend_wave_higher" in f) and ("09-24" in f or "09-26" in f) and ("2xArm+1xStanding" not in f)])
    # print()
    # print(len([f for f in has_gif_folder_list if ("both_arms_out" in f or "left_arm_out" in f or "right_arm_out" in f or "left_arm_extend_wave_higher" in f or "right_arm_extend_wave_higher" in f) and ("09-24" in f or "09-26" in f) and ("2xArm+1xStanding" not in f)]))
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


if __name__ == "__main__":
    folder_path = "/share/portal/hw575/CrossQ/train_logs"

    has_gif_folder_list = []

    for f in os.listdir(folder_path):
        has_gif = clean_folder(os.path.join(folder_path, f))
        
        if has_gif:
            has_gif_folder_list.append(os.path.join(folder_path, f, "eval"))

    print(has_gif_folder_list)
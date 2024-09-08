import matplotlib.pyplot as plt

import os

from tqdm import tqdm

# path = "/share/portal/aw588/examples/v3_seq_ex"
path = "/share/portal/aw588/examples/v3_random_joints_ex"


debug_path = "/share/portal/hw575/CrossQ/debugging/eval_finetune_data"
data_type = path.split("/")[-1]
debug_save_path = os.path.join(debug_path, data_type)
os.makedirs(debug_save_path, exist_ok=True)

# image files are of format: 0_pose.png
# list all the image files in path/anchor
anchor_files = [f for f in os.listdir(path + "/anchor") if f.endswith('.png')]

skip = 20

for i in tqdm(range(0, len(anchor_files), skip)):
    anchor_fp = os.path.join(path, "anchor", anchor_files[i])

    # Read the anchor image
    anchor_img = plt.imread(anchor_fp)

    for j in range(3):
        if data_type == "v3_seq_ex":
            anchor_base_name = anchor_files[i].split("_anchor_")[0]

            # Find the corresponding image file in path/pos that also begin with anchor_base_name
            #   path/pos has format: {anchor_base_name}_pos_frame{int}.png
            #   path/neg has format: {anchor_base_name}_neg_frame{int}.png
            # but we don't know what the frame number is
            pos_fp = os.path.join(path, "pos", [f for f in os.listdir(path + "/pos") if f.startswith(anchor_base_name)][0])
            neg_fp = os.path.join(path, "neg", [f for f in os.listdir(path + "/neg") if f.startswith(anchor_base_name)][0])
        else:
            # Find the corresponding image file in path/pos of format: 0_1_step.png, 0_2_pose.png, or 0_3_pose.png
            #   path/neg has format: 0_1_pose.png, 0_2_pose.png, or 0_3_pose.png
            if j == 0:
                pos_fp = os.path.join(path, "pos", f"{i}_{j+1}_step.png")
                neg_fp = os.path.join(path, "neg", f"{i}_{j+1}_pose.png")
            else:
                pos_fp = os.path.join(path, "pos", f"{i}_{j+1}_pose.png")
                neg_fp = os.path.join(path, "neg", f"{i}_{j+1}_pose.png")

        # Read the pos and neg images
        pos_img = plt.imread(pos_fp)
        neg_img = plt.imread(neg_fp)

        # Display the images
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(anchor_img)
        axes[0].axis("off")

        axes[1].imshow(pos_img)
        axes[1].axis("off")

        axes[2].imshow(neg_img)
        axes[2].axis("off")

        plt.suptitle(f"Iter={i}, sample={j}, {data_type}, (anc, pos, neg)")
        plt.tight_layout()
        plt.savefig(os.path.join(debug_save_path, f"{i}_{j}_eval.png"))

        plt.clf()
        plt.close(fig)

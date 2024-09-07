import copy
import numpy as np
import random


def set_seed(seed):
    """
    :param seed: An integer representing the seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    return seed

# def set_torch_seed(seed):
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)


# Perturb a joint configuration slightly to create a positive example
def perturb_joints_positively(init_qpos, joint_to_change, poses_thres, perc=0.3):
    new_qpos = copy.deepcopy(init_qpos)
    perc_variation = perc
    for idx in joint_to_change:
        std_dev = poses_thres[str(idx)]['std_dev']
        # randomly sample a float between 0 and perc
        # perc_variation = random.uniform(0, perc)
        # perturbation = np.random.uniform(-perc_variation * std_dev, perc_variation * std_dev)
        perturbation = perc * std_dev
        new_qpos[int(idx)] = init_qpos[int(idx)] + perturbation
    return new_qpos

# Perturb a joint configuration significantly to create a negative example
def perturb_joints_negatively(init_qpos, joint_to_change, poses_thres, perc=0.8):
    new_qpos = copy.deepcopy(init_qpos)
    for idx in joint_to_change:
        std_dev = poses_thres[str(idx)]['std_dev']
        # randomly sample a float between 0 and perc
        perturbation = np.random.uniform(1 * std_dev, 2 * std_dev)  # Using 100% to 200% for illustration
        # Randomly decide to add or subtract this perturbation
        perturbation *= np.random.choice([-1, 1])
        new_qpos[int(idx)] = init_qpos[int(idx)] + perturbation
    return new_qpos

def set_joints(joint_config, init_qpos):
    new_qpos = copy.deepcopy(init_qpos)
    for idx in joint_config.keys():
        new_qpos[int(idx)] = joint_config[int(idx)] # z-coordinate of torso
    return new_qpos

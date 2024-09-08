import copy
import numpy as np
import random

from loguru import logger

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


"""++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Body Distortion Helper Functions
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"""
def generate_body_distortion_arm_config():
    """
    Generates a random pose configuration for the arm joints. This is used to set the anchor pose.
    
    Note:
        We are only randomly generating initial arm joint configurations.
        
    Return
        selected_joints (list): A list of randomly selected joint indices that got changed
        pose_config (dict): A dictionary with random joint indices and their corresponding radian values.
    """
    # Hard code arm related joints
    arm_joints = [18, 19, 20, 21, 22, 23]

    # Random decide how many joints to modify (at least 1)
    num_joints_to_change = random.randint(1, len(arm_joints))

    # Randomly select the joint indices to change
    selected_joints = random.sample(arm_joints, num_joints_to_change)

    # Generate random radian values for these joints
    # Assuming radian values should be between -pi and pi
    pose_config = {joint: np.random.uniform(-np.pi, np.pi) for joint in selected_joints}

    # Land the entire torso on the floor
    pose_config[2] = 1.3

    logger.debug(f"Selected joints: {selected_joints}")
    logger.debug(f"Pose config: {pose_config}")

    return selected_joints, pose_config

def mild_body_distortion(init_qpos):
    """
    Given an initial qpos, mildly distort the body and change the arm configuration slightly.

    Return:
        new_qpos (list): A list of new qpos values
    """
    new_qpos = copy.deepcopy(init_qpos)
    
    body_joints = [7, 8, 9]

    # Random decide how many joints to modify (at least 1)
    num_joints_to_change = random.randint(1, len(body_joints))

    # Randomly select the joint indices to change
    selected_joints = random.sample(body_joints, num_joints_to_change)

    logger.debug(f"Selected joints: {selected_joints}")

    # Changing 9 (x-angle of the abdomen (in pelvis)) is tricky because to adjust it, we need to change 4, which also affects the arm position
    # For now, we don't handle arm position. We just adjust the legs
    joint_7_divisor = 4
    joint_8_divisor = 4
    joint_9_divisor = 10

    if 9 in selected_joints:
        new_qpos[9] = np.random.uniform(-np.pi/joint_9_divisor, np.pi/joint_9_divisor)
        new_qpos[4] = - new_qpos[9]
        # Bring the legs back close to the original position
        new_qpos[10] = (new_qpos[9] - np.pi/joint_9_divisor/4)
        new_qpos[14] = -(new_qpos[9] - np.pi/joint_9_divisor/4)
        
        # To make the pose not too distorted, we reduce joint 7's divisor
        joint_7_divisor /= 2

    # If we are changing 7 (z-angle of the abdomen (in lower_waist))
    #  the maximum range should be between -pi/3 and pi/3
    if 7 in selected_joints:
        new_qpos[7] = np.random.uniform(-np.pi/joint_7_divisor, np.pi/joint_7_divisor)
        logger.debug(f"New qpos[7]: {new_qpos[7]}")

    # If we are changing 8 (y-angle of the abdomen (in lower_waist))
    #   the maximum range should be between -pi/4 and pi/4
    # We also have to make 12 and 16 be in the same direction
    #   their value should be based on the value of 8, but they can + or - pi/4/4
    if 8 in selected_joints:
        new_qpos[8] = np.random.uniform(-np.pi/joint_8_divisor, np.pi/joint_8_divisor)
        # Bring the legs back close to the original position
        new_qpos[12] = - new_qpos[8] + (-1)**np.random.randint(0, 2) * np.pi/joint_8_divisor/3
        new_qpos[16] = - new_qpos[8] + (-1)**np.random.randint(0, 2) * np.pi/joint_8_divisor/3
        logger.debug(f"New qpos[8]: {new_qpos[8]}")
        logger.debug(f"New qpos[12]: {new_qpos[12]}")
        logger.debug(f"New qpos[16]: {new_qpos[16]}")

    # Hard code arm related joints
    arm_joints = [18, 19, 20, 21, 22, 23]

    # Random decide how many joints to modify (at least 1, at most half of the arm joints)
    num_joints_to_change = random.randint(0, len(arm_joints)//2)

    # Randomly select the joint indices to change
    selected_joints = random.sample(arm_joints, num_joints_to_change)

    arm_joint_divisor = 8

    for idx in selected_joints:
        # Randomly decide to add or subtract this perturbation
        new_qpos[idx] += (-1)**np.random.randint(0, 2) * np.random.uniform(-np.pi/arm_joint_divisor, np.pi/arm_joint_divisor)

    return new_qpos
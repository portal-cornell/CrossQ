import numpy as np
import random


def generate_random_pose_config(num_joints=19, joint_ranges=[(2, 4), (7, 23)]):
    """
    Generates a random pose configuration for a range of joints.
    
    Parameters:
        num_joints (int): Maximum number of joints to potentially adjust.
        joint_range (tuple): The range of joint indices to consider.
    
    Returns:
        dict: A dictionary with random joint indices and their corresponding radian values.
    """
    # Create a list of all possible joint indices
    joints_indices = []
    for start, end in joint_ranges:
        joints_indices.extend(range(start, end + 1))
    
    # Remove joints 5 and 6 if they're in the list
    joints_indices = [j for j in joints_indices if j not in (5, 6)]
    
    # Randomly decide how many joints to modify (at least 1)
    num_joints_to_change = random.randint(1, num_joints)
    
    # Randomly select the joint indices to change
    selected_joints = random.sample(joints_indices, num_joints_to_change)
    
    # Generate random radian values for these joints
    # Assuming radian values should be between -pi and pi
    pose_config = {joint: np.random.uniform(-np.pi, np.pi) for joint in selected_joints}

    if 2 in pose_config:
        # 1.3 is when the humanoid's feet are touching the ground when it's standing
        # above 2.2 is when the humanoid is about to be out of the frame
        pose_config[2] = np.random.uniform(1.3, 2.2)
    
    return selected_joints, pose_config
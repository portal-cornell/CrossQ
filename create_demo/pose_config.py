import numpy as np

pose_config_dict = {
    "init-pose": {},
    "both-arms-out":
    {
        2: 1.3, # Land the entire torso on the floor
        18: -np.pi/4, # Right upper arm perpendicular to the side of the torso
        19: -np.pi*3/16, # Left upper arm align with the side of torso
        20: -np.pi/8*5, # Make the angle between right upper arm and right lower arm close to pi
        21: np.pi/4, # Left upper arm perpendicular to the side of the torso
        22: np.pi*3/16, # Left upper arm align with the side of torso
        23: -np.pi/8*5, # Make the angle between left upper arm and left lower arm close to pi
    },
    "right-arm-out":
    {
        2: 1.3,
        18: -np.pi/4, # Right upper arm perpendicular to the side of the torso
        19: -np.pi*3/16, # Left upper arm align with the side of torso
        20: -np.pi*5/8, # Make the angle between right upper arm and right lower arm close to pi
        21: -np.pi*3/16, # Left upper arm perpendicular to the side of the torso
        22: np.pi*3/16, # Left upper arm align with the side of torso
        23: -np.pi*5/8, # Make the angle between left upper arm and left lower arm close to pi
    },
    "left-arm-out":
    {
        2: 1.3, # Land the entire torso on the floor
        18: np.pi*3/16, # Right upper arm perpendicular to the side of the torso
        19: -np.pi*3/16, # Left upper arm align with the side of torso
        20: -np.pi*5/8, # Make the angle between right upper arm and right lower arm close to pi
        21: np.pi/4, # Left upper arm perpendicular to the side of the torso
        22: np.pi*3/16, # Left upper arm align with the side of torso
        23: -np.pi*5/8, # Make the angle between left upper arm and left lower arm close to pi
    }
}
import numpy as np

pose_config_dict = {
    "init-pose": {},
    "both-arms-out":
    {
        2: 1.3, # Land the entire torso on the floor
        18: -np.pi/4, # [R] Right upper arm perpendicular to the side of the torso
        19: -np.pi*3/16, # [R] Right upper arm align with the side of torso
        20: -np.pi/8*5, # [R] Make the angle between right upper arm and right lower arm close to pi
        21: np.pi/4, # [L] Left upper arm perpendicular to the side of the torso
        22: np.pi*3/16, # [L] Left upper arm align with the side of torso
        23: -np.pi/8*5, # [L] Make the angle between left upper arm and left lower arm close to pi
    },
    "right-arm-out":
    {
        2: 1.3,
        18: -np.pi/4, # [R] Right upper arm perpendicular to the side of the torso
        19: -np.pi*3/16, # [R] Right upper arm align with the side of torso
        20: -np.pi*5/8, # [R] Make the angle between right upper arm and right lower arm close to pi
        21: -np.pi*3/16, # [L] Left upper arm perpendicular to the side of the torso
        22: np.pi*3/16, # [L] Left upper arm align with the side of torso
        23: -np.pi*5/8, # [L] Make the angle between left upper arm and left lower arm close to pi
    },
    "left-arm-out":
    {
        2: 1.3, # Land the entire torso on the floor
        18: np.pi*3/16, # [R] Right upper arm perpendicular to the side of the torso
        19: -np.pi*3/16, # [R] Right upper arm align with the side of torso
        20: -np.pi*5/8, # [R] Make the angle between right upper arm and right lower arm close to pi
        21: np.pi/4, # [L] Left upper arm perpendicular to the side of the torso
        22: np.pi*3/16, # [L] Left upper arm align with the side of torso
        23: -np.pi*5/8, # [L] Make the angle between left upper arm and left lower arm close to pi
    },
    "left-arm-extend-wave-lower":
    {
        2: 1.3, # Land the entire torso on the floor
        18: np.pi*3/16, # [R] Right upper arm perpendicular to the side of the torso
        19: -np.pi*3/16, # [R] Left upper arm align with the side of torso
        20: -np.pi*5/8, # [R] Make the angle between right upper arm and right lower arm close to pi
        21: np.pi*3/8, # [L] Left upper arm perpendicular to the side of the torso
        22: -np.pi/12, # [L] Left upper arm align with the side of torso
        23: -np.pi*5/8, # [L] Make the angle between left upper arm and left lower arm close to pi
    },
    "left-arm-extend-wave-higher":
    {
        2: 1.3, # Land the entire torso on the floor
        18: np.pi*3/16, # [R] Right upper arm perpendicular to the side of the torso
        19: -np.pi*3/16, # [R] Left upper arm align with the side of torso
        20: -np.pi*5/8, # [R] Make the angle between right upper arm and right lower arm close to pi
        21: np.pi*5/8, # [L] Left upper arm perpendicular to the side of the torso
        22: -np.pi/12, # [L] Left upper arm align with the side of torso
        23: -np.pi*5/8, # [L] Make the angle between left upper arm and left lower arm close to pi
    },
    "right-arm-extend-wave-lower":
    {
        2: 1.3, # Land the entire torso on the floor
        18: -np.pi*3/8, # [R] Right upper arm perpendicular to the side of the torso
        19: np.pi/12, # [R] Left upper arm align with the side of torso
        20: -np.pi*5/8, # [R] Make the angle between right upper arm and right lower arm close to pi
        21: -np.pi*3/16, # [L] Left upper arm perpendicular to the side of the torso
        22: np.pi*3/16, # [L] Left upper arm align with the side of torso
        23: -np.pi*5/8, # [L] Make the angle between left upper arm and left lower arm close to pi
    },
    "right-arm-extend-wave-higher":
    {
        2: 1.3, # Land the entire torso on the floor
        18: -np.pi*5/8, # [R] Right upper arm perpendicular to the side of the torso
        19: np.pi/12, # [R] Left upper arm align with the side of torso
        20: -np.pi*5/8, # [R] Make the angle between right upper arm and right lower arm close to pi
        21: -np.pi*3/16, # [L] Left upper arm perpendicular to the side of the torso
        22: np.pi*3/16, # [L] Left upper arm align with the side of torso
        23: -np.pi*5/8, # [L] Make the angle between left upper arm and left lower arm close to pi
    },
    "arms_bracket_right":
    {
        2: 1.3, # Land the entire torso on the floor
        18: -np.pi/4, # [R] Right upper arm perpendicular to the side of the torso
        19: np.pi/12, # [R] Right upper arm align with the side of torso
        20: -np.pi*5/8, # [R] Make the angle between right upper arm and right lower arm close to pi
        21: -np.pi/3, # [L] Left upper arm perpendicular to the side of the torso
        22: -np.pi*5/16, # [L] Left upper arm align with the side of torso
        23: -np.pi*5/8, # [L] Make the angle between left upper arm and left lower arm close to pi
    },
    "arms_bracket_left":
    {
        2: 1.3, # Land the entire torso on the floor
        18: np.pi/3, # [R] Right upper arm perpendicular to the side of the torso
        19: np.pi*5/16, # [R] Right upper arm align with the side of torso
        20: -np.pi*5/8, # [R] Make the angle between right upper arm and right lower arm close to pi
        21: np.pi/4, # [L] Left upper arm perpendicular to the side of the torso
        22: -np.pi/12, # [L] Left upper arm align with the side of torso
        23: -np.pi*5/8, # [L] Make the angle between left upper arm and left lower arm close to pi
    },
    "arms_bracket_up":
    {
        2: 1.3, # Land the entire torso on the floor
        18: -np.pi*6/8, # [R] Right upper arm perpendicular to the side of the torso
        19: np.pi/12, # [R] Right upper arm align with the side of torso
        20: -np.pi*5/8, # [R] Make the angle between right upper arm and right lower arm close to pi
        21: np.pi*6/8, # [L] Left upper arm perpendicular to the side of the torso
        22: -np.pi/12, # [L] Left upper arm align with the side of torso
        23: -np.pi*5/8, # [L] Make the angle between left upper arm and left lower arm close to pi
    },
    "arms_bracket_down":
    {
        2: 1.3, # Land the entire torso on the floor
        18: np.pi/4, # [R] Right upper arm perpendicular to the side of the torso
        19: -np.pi*3/16, # [R] Right upper arm align with the side of torso
        20: -np.pi*5/8, # [R] Make the angle between right upper arm and right lower arm close to pi
        21: -np.pi/4, # [L] Left upper arm perpendicular to the side of the torso
        22: np.pi*3/16, # [L] Left upper arm align with the side of torso
        23: -np.pi*5/8, # [L] Make the angle between left upper arm and left lower arm close to pi
    },
    "arms_crossed_high":
    {
        2: 1.3, # Land the entire torso on the floor
        # 18: -np.pi*5/8, # [R] Right upper arm perpendicular to the side of the torso
        19: np.pi*7/16, # [R] Right upper arm align with the side of torso
        20: -np.pi*3/8, # [R] Make the angle between right upper arm and right lower arm close to pi
        # 21: np.pi*5/8, # [L] Left upper arm perpendicular to the side of the torso
        22: -np.pi*6/16, # [L] Left upper arm align with the side of torso
        23: -np.pi*3/8, # [L] Make the angle between left upper arm and left lower arm close to pi
    },
    "default-but-arms-up":
    {
        2: 1.3, # Land the entire torso on the floor
        18: -np.pi/4, # [R] Right upper arm perpendicular to the side of the torso
        # 19: np.pi*3/16, # [R] Right upper arm align with the side of torso
        # 20: -np.pi*5/8, # [R] Make the angle between right upper arm and right lower arm close to pi
        21: np.pi/4, # [L] Left upper arm perpendicular to the side of the torso
        # 22: -np.pi*3/16, # [L] Left upper arm align with the side of torso
        # 23: -np.pi*5/8, # [L] Make the angle between left upper arm and left lower arm close to pi
    }
}
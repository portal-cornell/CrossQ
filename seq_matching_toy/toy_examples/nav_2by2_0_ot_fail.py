import numpy as np

"""
Simple example to show ot fails compared to dtw
"""
nav_2by2_0_ot_fail = {
    "plot": {
        "reward_vmin": 0,
        "reward_vmax": 2,
    },
    "starting_pos": np.array([0, 0]),
    "map_array": np.array([
        [0, 0],
        [0, 0],
    ]),
    "ref_seq": [
        np.array([[0, 1], 
                  [0, 0]]),
        np.array([[0, 0],
                  [1, 0]]),
    ],
    "obs_seqs": {
        0: {
            "descriptions": "Correct",
            "seq": [
                np.array([[0, 1], 
                          [0, 0]]),
                np.array([[0, 0], 
                          [0, 1]]),
                np.array([[0, 0], 
                          [1, 0]]),
            ]
        },
        1: {
            "descriptions": "Inverse. OT gives equally high reward",
            "seq": [
                np.array([[0, 0], 
                          [1, 0]]),
                np.array([[0, 0], 
                          [0, 1]]),
                np.array([[0, 1], 
                          [0, 0]]),
            ]
        }
    }
}
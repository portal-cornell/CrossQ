import numpy as np

"""
Simple example to show dtw fails compared to sdtw
"""
nav_1by4_0_dtw_fail = {
    "plot": {
        "reward_vmin": 0,
        "reward_vmax": 2,
    },
    "starting_pos": np.array([0, 0]),
    "map_array": np.array([
        [0, 0, 0, 0],
    ]),
    "ref_seq": [
        np.array([[1, 0, 0, 0]]),
        np.array([[0, 0, 0, 1]]),
    ],
    "obs_seqs": {
        0: {
            "descriptions": "Correct",
            "seq": [
                np.array([[1, 0, 0, 0]]),
                np.array([[0, 1, 0, 0]]),
                np.array([[0, 0, 1, 0]]),
                np.array([[0, 0, 0, 1]]),
            ]
        },
        1: {
            "descriptions": "Stuck at ref seq frame 1",
            "seq": [
                np.array([[1, 0, 0, 0]]),
                np.array([[1, 0, 0, 0]]),
                np.array([[1, 0, 0, 0]]),
                np.array([[0, 1, 0, 0]]),
            ]
        }
    }
}
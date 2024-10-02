import numpy as np

nav_3by3_1 = {
    "plot": {
        "reward_vmin": -4,
        "reward_vmax": 0,
    },
    "starting_pos": np.array([0, 0]),
    "map_array": np.array([
        [0, 0, 0],
        [0, -1, 0],
        [0, 0, 0],
    ]),
    "ref_seq": [
        np.array([[0, 0, 0], 
                  [1, -1, 0],
                  [0, 0, 0]]),
        np.array([[0, 0, 0], 
                  [0, -1, 1],
                  [0, 0, 0]]),
    ],
    "obs_seqs": {
        0: {
            "descriptions": "Hypothesis: DTW will get stuck at the first idx",
            "seq": [
                np.array([[0, 0, 0], 
                          [1, -1, 0],
                          [0, 0, 0]]),
                np.array([[0, 0, 0], 
                          [1, -1, 0],
                          [0, 0, 0]]),
                np.array([[0, 0, 0], 
                          [1, -1, 0],
                          [0, 0, 0]]),
                np.array([[0, 0, 0], 
                          [1, -1, 0],
                          [0, 0, 0]]),
                np.array([[0, 0, 0], 
                          [1, -1, 0],
                          [0, 0, 0]]),
            ],
        },
        1: {
            "descriptions": "The correct rollout",
            "seq": [
                np.array([[0, 0, 0], 
                          [1, -1, 0],
                          [0, 0, 0]]),
                np.array([[0, 0, 0], 
                          [0, -1, 0],
                          [1, 0, 0]]),
                np.array([[0, 0, 0], 
                          [0, -1, 0],
                          [0, 1, 0]]),
                np.array([[0, 0, 0], 
                          [0, -1, 0],
                          [0, 0, 1]]),
                np.array([[0, 0, 0], 
                          [0, -1, 1],
                          [0, 0, 0]]),
            ],
        }
    }
}
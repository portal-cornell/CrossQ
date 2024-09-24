import numpy as np

nav_3by3_0_full_ref = {
    "plot": {
        "reward_vmin": -4,
        "reward_vmax": 0,
    },
    "map_array": np.array([
        [0, 0, 0],
        [0, -1, 0],
        [0, 0, 0],
    ]),
    "ref_seq": [
                np.array([[1, 0, 0], 
                          [0, -1, 0],
                          [0, 0, 0]]),
                np.array([[0, 1, 0], 
                          [0, -1, 0],
                          [0, 0, 0]]),
                np.array([[0, 0, 1], 
                          [0, -1, 0],
                          [0, 0, 0]]),
                np.array([[0, 0, 0], 
                          [0, -1, 1],
                          [0, 0, 0]]),
                np.array([[0, 0, 0], 
                          [0, -1, 0],
                          [0, 0, 1]]),
                np.array([[0, 0, 0], 
                          [0, -1, 0],
                          [0, 1, 0]]),
                np.array([[0, 0, 0], 
                          [0, -1, 0],
                          [1, 0, 0]]),
                np.array([[0, 0, 0], 
                          [1, -1, 0],
                          [0, 0, 0]]),
            ],
    "obs_seqs": {
        0: {
            "descriptions": "Hypothesis: OT will be happy to reach the 2nd idx before 1st idx",
            "seq": [
                np.array([[1, 0, 0], 
                          [0, -1, 0],
                          [0, 0, 0]]),
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
                          [0, -1, 0],
                          [0, 0, 1]]),
                np.array([[0, 0, 0], 
                          [0, -1, 0],
                          [0, 0, 1]]),
                np.array([[0, 0, 0], 
                          [0, -1, 0],
                          [0, 0, 1]]),
            ],
        },
        1: {
            "descriptions": "The correct rollout",
            "seq": [
                np.array([[1, 0, 0], 
                          [0, -1, 0],
                          [0, 0, 0]]),
                np.array([[0, 1, 0], 
                          [0, -1, 0],
                          [0, 0, 0]]),
                np.array([[0, 0, 1], 
                          [0, -1, 0],
                          [0, 0, 0]]),
                np.array([[0, 0, 0], 
                          [0, -1, 1],
                          [0, 0, 0]]),
                np.array([[0, 0, 0], 
                          [0, -1, 0],
                          [0, 0, 1]]),
                np.array([[0, 0, 0], 
                          [0, -1, 0],
                          [0, 1, 0]]),
                np.array([[0, 0, 0], 
                          [0, -1, 0],
                          [1, 0, 0]]),
                np.array([[0, 0, 0], 
                          [1, -1, 0],
                          [0, 0, 0]]),
            ],
        }
    }
}
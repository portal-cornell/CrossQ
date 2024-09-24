import numpy as np

"""
A slightly easier version of nav_3by3_0. 
    1st and 2nd goal are equally far from the start (so hopefully dtw gets stuck less than ot)
"""
nav_3by3_2 = {
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
        np.array([[0, 0, 0], 
                  [0, -1, 1],
                  [0, 0, 0]]),
        np.array([[0, 0, 0], 
                  [0, -1, 0],
                  [0, 1, 0]]),
    ],
    "obs_seqs": {
        0: {
            "descriptions": "Verify using shortest path distance (A*) as the cost",
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
                          [0, 1, 0]]),
                np.array([[0, 0, 0], 
                          [0, -1, 0],
                          [0, 1, 0]]),
                np.array([[0, 0, 0], 
                          [0, -1, 0],
                          [0, 1, 0]]),
                np.array([[0, 0, 0], 
                          [0, -1, 0],
                          [0, 1, 0]]),
            ]
        }
    }
}
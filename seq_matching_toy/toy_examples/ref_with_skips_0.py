import numpy as np

ref_with_skips_0 = {
    "plot": {
        "reward_vmin": -2,
        "reward_vmax": 0,
    },
    "ref_seq": [
        np.array([[0, 0], 
                  [1, 0]]),
        np.array([[1, 0], 
                  [0, 0]]),
        np.array([[0, 1], 
                  [0, 0]]),
    ],
    "obs_seqs": {
        0: {
            "descriptions": "H: A good sequence should create a matching that evenly distribute across the ref_seq",
            "seq": [
                np.array([[0, 0], 
                          [0, 0]]),
                np.array([[0, 0], 
                          [1, 0]]),
                np.array([[0, 0], 
                          [0, 0]]),
                np.array([[1, 0], 
                          [0, 0]]),
                np.array([[0, 0], 
                          [0, 0]]),
                np.array([[0, 1], 
                          [0, 0]]),
            ],
        },
        1: {
            "descriptions": "Will get stuck at the 1st idx",
            "seq": [
                np.array([[0, 0], 
                          [0, 0]]),
                np.array([[0, 0], 
                          [1, 0]]),
                np.array([[0, 0], 
                          [1, 0]]),
                np.array([[0, 0], 
                          [1, 0]]),
                np.array([[0, 0], 
                          [0, 0]]),
                np.array([[0, 1], 
                          [0, 0]]),
            ],
        },
        2: {
            "descriptions": "Will get stuck at the 2nd idx",
            "seq": [
                np.array([[0, 0], 
                          [0, 0]]),
                np.array([[0, 1], 
                          [0, 0]]),
                np.array([[0, 1], 
                          [0, 0]]),
                np.array([[0, 1], 
                          [0, 0]]),
                np.array([[0, 1], 
                          [0, 0]]),
                np.array([[0, 1], 
                          [0, 0]]),
            ],
        }
    }
}
import numpy as np

criss_cross_0 = {
    "plot": {
        "reward_vmin": -2,
        "reward_vmax": 0,
    },
    "ref_seq": [
        np.array([[0, 0], 
                  [0, 0]]),
        np.array([[0, 1], 
                  [0, 0]]),
        np.array([[0, 1], 
                  [1, 0]]),
        np.array([[0, 0], 
                  [1, 0]]),
    ],
    "obs_seqs": {
        0: {
            "descriptions": "Hypothesis: OT will flip matching idx 1 and 3",
            "seq": [
                np.array([[0, 0], 
                          [0, 0]]),
                np.array([[0, 0], 
                          [1, 0]]),
                np.array([[0, 1], 
                          [1, 0]]),
                np.array([[0, 1], 
                          [0, 0]]),
            ],
        },
        1: {
            "descriptions": "Not sure what will happen if the sequence stays at the 1st idx",
            "seq": [
                np.array([[0, 0], 
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
import numpy as np
nav_backwards_easier = {
    "plot": {
        "reward_vmin": -6,
        "reward_vmax": 0,
    },
    "starting_pos": np.array([1, 0]),
    "map_array": 
        np.array([[0, 0, 0, 0, 0],
                  [0, -1, -1, -1, 0],
                  [0, 0, 0, 0, 0]]),
    "ref_seq": [
        np.array([[0, 0, 0, 1, 0],
                    [0, -1, -1, -1, 0],
                    [0, 0, 0, 0, 0]]),
        np.array([[0, 0, 0, 0, 0],
                    [0, -1, -1, -1, 1],
                    [0, 0, 0, 0, 0]]),
    ],
    "obs_seqs": {
        0: {
            "descriptions": "Correct",
            "seq": [
                np.array([[1, 0, 0, 0, 0],
                          [0, -1, -1, -1, 0],
                          [0, 0, 0, 0, 0]]),
                np.array([[0, 1, 0, 0, 0],
                          [0, -1, -1, -1, 0],
                          [0, 0, 0, 0, 0]]),
                np.array([[0, 0, 1, 0, 0],
                          [0, -1, -1, -1, 0],
                          [0, 0, 0, 0, 0]]),
                np.array([[0, 0, 0, 1, 0],
                          [0, -1, -1, -1, 0],
                          [0, 0, 0, 0, 0]]),
                np.array([[0, 0, 0, 0, 1],
                          [0, -1, -1, -1, 0],
                          [0, 0, 0, 0, 0]]),
                np.array([[0, 0, 0, 0, 0],
                          [0, -1, -1, -1, 1],
                          [0, 0, 0, 0, 0]]),
                np.array([[0, 0, 0, 0, 0],
                          [0, -1, -1, -1, 0],
                          [0, 0, 0, 0, 1]]),
                np.array([[0, 0, 0, 0, 0],
                          [0, -1, -1, -1, 0],
                          [0, 0, 0, 1, 0]]),
                np.array([[0, 0, 0, 0, 0],
                          [0, -1, -1, -1, 0],
                          [0, 0, 1, 0, 0]]),
                np.array([[0, 0, 0, 0, 0],
                          [0, -1, -1, -1, 0],
                          [0, 1, 0, 0, 0]]),
                np.array([[0, 0, 0, 0, 0],
                          [0, -1, -1, -1, 0],
                          [1, 0, 0, 0, 0]]),
                np.array([[0, 0, 0, 0, 0],
                          [1, -1, -1, -1, 0],
                          [0, 0, 0, 0, 0]]),
                np.array([[1, 0, 0, 0, 0],
                          [0, -1, -1, -1, 0],
                          [0, 0, 0, 0, 0]]),
                np.array([[0, 1, 0, 0, 0],
                          [0, -1, -1, -1, 0],
                          [0, 0, 0, 0, 0]]),
                np.array([[0, 0, 1, 0, 0],
                          [0, -1, -1, -1, 0],
                          [0, 0, 0, 0, 0]]),
                np.array([[0, 0, 0, 1, 0],
                          [0, -1, -1, -1, 0],
                          [0, 0, 0, 0, 0]]),
            ]
        },
        1: {
            "descriptions": "Reversed",
                        "seq": [
                np.array([[1, 0, 0, 0, 0],
                          [0, -1, -1, -1, 0],
                          [0, 0, 0, 0, 0]]),
                np.array([[0, 1, 0, 0, 0],
                          [0, -1, -1, -1, 0],
                          [0, 0, 0, 0, 0]]),
                np.array([[0, 0, 1, 0, 0],
                          [0, -1, -1, -1, 0],
                          [0, 0, 0, 0, 0]]),
                np.array([[0, 0, 0, 1, 0],
                          [0, -1, -1, -1, 0],
                          [0, 0, 0, 0, 0]]),
                np.array([[0, 0, 1, 0, 0],
                          [0, -1, -1, -1, 0],
                          [0, 0, 0, 0, 0]]),
                np.array([[0, 1, 0, 0, 0],
                          [0, -1, -1, -1, 0],
                          [0, 0, 0, 0, 0]]),
                np.array([[1, 0, 0, 0, 0],
                          [0, -1, -1, -1, 0],
                          [0, 0, 0, 0, 0]]),
                np.array([[0, 1, 0, 0, 0],
                          [0, -1, -1, -1, 0],
                          [0, 0, 0, 0, 0]]),
                np.array([[0, 0, 1, 0, 0],
                          [0, -1, -1, -1, 0],
                          [0, 0, 0, 0, 0]]),
                np.array([[0, 0, 0, 1, 0],
                          [0, -1, -1, -1, 0],
                          [0, 0, 0, 0, 0]]),
                np.array([[0, 0, 0, 0, 1],
                          [0, -1, -1, -1, 0],
                          [0, 0, 0, 0, 0]]),
                np.array([[0, 0, 0, 0, 0],
                          [0, -1, -1, -1, 1],
                          [0, 0, 0, 0, 0]]),
                np.array([[0, 0, 0, 0, 0],
                          [0, -1, -1, -1, 0],
                          [0, 0, 0, 0, 1]]),
                np.array([[0, 0, 0, 0, 0],
                          [0, -1, -1, -1, 0],
                          [0, 0, 0, 1, 0]]),
                np.array([[0, 0, 0, 0, 0],
                          [0, -1, -1, -1, 0],
                          [0, 0, 1, 0, 0]]),
                np.array([[0, 0, 0, 0, 0],
                          [0, -1, -1, -1, 0],
                          [0, 1, 0, 0, 0]])
            ]
        },

        3: {
            "descriptions": "Stuck at the first pose",
            "seq": [
                np.array([[1, 0, 0, 0, 0],
                          [0, -1, -1, -1, 0],
                          [0, 0, 0, 0, 0]]),
                np.array([[0, 1, 0, 0, 0],
                          [0, -1, -1, -1, 0],
                          [0, 0, 0, 0, 0]]),
                np.array([[0, 0, 1, 0, 0],
                          [0, -1, -1, -1, 0],
                          [0, 0, 0, 0, 0]]),
                np.array([[0, 0, 0, 1, 0],
                          [0, -1, -1, -1, 0],
                          [0, 0, 0, 0, 0]]),
                np.array([[0, 0, 1, 0, 0],
                          [0, -1, -1, -1, 0],
                          [0, 0, 0, 0, 0]]),
                np.array([[0, 1, 0, 0, 0],
                          [0, -1, -1, -1, 0],
                          [0, 0, 0, 0, 0]]),
                np.array([[1, 0, 0, 0, 0],
                          [0, -1, -1, -1, 0],
                          [0, 0, 0, 0, 0]]),
                np.array([[1, 0, 0, 0, 0],
                          [0, -1, -1, -1, 0],
                          [0, 0, 0, 0, 0]]),
                np.array([[1, 0, 0, 0, 0],
                          [0, -1, -1, -1, 0],
                          [0, 0, 0, 0, 0]]),
                np.array([[1, 0, 0, 0, 0],
                          [0, -1, -1, -1, 0],
                          [0, 0, 0, 0, 0]]),
                np.array([[1, 0, 0, 0, 0],
                          [0, -1, -1, -1, 0],
                          [0, 0, 0, 0, 0]]),
                np.array([[1, 0, 0, 0, 0],
                          [0, -1, -1, -1, 0],
                          [0, 0, 0, 0, 0]]),
                np.array([[1, 0, 0, 0, 0],
                          [0, -1, -1, -1, 0],
                          [0, 0, 0, 0, 0]]),
                np.array([[1, 0, 0, 0, 0],
                          [0, -1, -1, -1, 0],
                          [0, 0, 0, 0, 0]])
            ]
        },
        4: {
            "descriptions": "Stuck at 1 + first pose",
            "seq": [
                np.array([[1, 0, 0, 0, 0],
                          [0, -1, -1, -1, 0],
                          [0, 0, 0, 0, 0]]),
                np.array([[0, 1, 0, 0, 0],
                          [0, -1, -1, -1, 0],
                          [0, 0, 0, 0, 0]]),
                np.array([[0, 0, 1, 0, 0],
                          [0, -1, -1, -1, 0],
                          [0, 0, 0, 0, 0]]),
                np.array([[0, 0, 0, 1, 0],
                          [0, -1, -1, -1, 0],
                          [0, 0, 0, 0, 0]]),
                np.array([[0, 0, 0, 0, 1],
                          [0, -1, -1, -1, 0],
                          [0, 0, 0, 0, 0]]),
                np.array([[0, 0, 0, 0, 1],
                          [0, -1, -1, -1, 0],
                          [0, 0, 0, 0, 0]]),
                np.array([[0, 0, 0, 0, 1],
                          [0, -1, -1, -1, 0],
                          [0, 0, 0, 0, 0]]),
                np.array([[0, 0, 0, 0, 1],
                          [0, -1, -1, -1, 0],
                          [0, 0, 0, 0, 0]]),
            ]
        },
        5: {
            "descriptions": "Stuck at 2 + first pose",
            "seq": [
                np.array([[1, 0, 0, 0, 0],
                          [0, -1, -1, -1, 0],
                          [0, 0, 0, 0, 0]]),
                np.array([[0, 1, 0, 0, 0],
                          [0, -1, -1, -1, 0],
                          [0, 0, 0, 0, 0]]),
                np.array([[0, 0, 1, 0, 0],
                          [0, -1, -1, -1, 0],
                          [0, 0, 0, 0, 0]]),
                np.array([[0, 0, 0, 1, 0],
                          [0, -1, -1, -1, 0],
                          [0, 0, 0, 0, 0]]),
                np.array([[0, 0, 0, 0, 1],
                          [0, -1, -1, -1, 0],
                          [0, 0, 0, 0, 0]]),
                np.array([[0, 0, 0, 0, 0],
                          [0, -1, -1, -1, 1],
                          [0, 0, 0, 0, 0]]),
                np.array([[0, 0, 0, 0, 0],
                          [0, -1, -1, -1, 1],
                          [0, 0, 0, 0, 0]]),
                np.array([[0, 0, 0, 0, 0],
                          [0, -1, -1, -1, 1],
                          [0, 0, 0, 0, 0]]),
            ]
        }
    }
}
import numpy as np

nav_3by3_0 = {
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
                  [0, -1, 0],
                  [0, 0, 1]]),
        np.array([[0, 0, 0], 
                  [1, -1, 0],
                  [0, 0, 0]]),
    ],
    "obs_seqs": {
        0: {
            "descriptions": "The correct rollout",
            "seq": [
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
        },
        1: {
            "descriptions": "1 step progress",
            "seq": [
                np.array([[0, 1, 0], 
                          [0, -1, 0],
                          [0, 0, 0]]),
                np.array([[0, 1, 0], 
                          [0, -1, 0],
                          [0, 0, 0]]),
                np.array([[0, 1, 0], 
                          [0, -1, 0],
                          [0, 0, 0]]),
                np.array([[0, 1, 0], 
                          [0, -1, 0],
                          [0, 0, 0]]),
                np.array([[0, 1, 0], 
                          [0, -1, 0],
                          [0, 0, 0]]),
                np.array([[0, 1, 0], 
                          [0, -1, 0],
                          [0, 0, 0]]),
                np.array([[0, 1, 0], 
                          [0, -1, 0],
                          [0, 0, 0]]),
            ]
        },
        2: {
            "description": "2 step progress",
            "seq": [
                np.array([[0, 1, 0], 
                          [0, -1, 0],
                          [0, 0, 0]]),
                np.array([[0, 0, 1], 
                          [0, -1, 0],
                          [0, 0, 0]]),
                np.array([[0, 0, 1], 
                          [0, -1, 0],
                          [0, 0, 0]]),
                np.array([[0, 0, 1], 
                          [0, -1, 0],
                          [0, 0, 0]]),
                np.array([[0, 0, 1], 
                          [0, -1, 0],
                          [0, 0, 0]]),
                np.array([[0, 0, 1], 
                          [0, -1, 0],
                          [0, 0, 0]]),
                np.array([[0, 0, 1], 
                          [0, -1, 0],
                          [0, 0, 0]]),
            ]
        },
        3: {
            "description": "3 step progress",
            "seq": [
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
                          [0, -1, 1],
                          [0, 0, 0]]),
                np.array([[0, 0, 0], 
                          [0, -1, 1],
                          [0, 0, 0]]),
                np.array([[0, 0, 0], 
                          [0, -1, 1],
                          [0, 0, 0]]),
                np.array([[0, 0, 0], 
                          [0, -1, 1],
                          [0, 0, 0]]),
            ]
        },
        4: {
            "description": "4 step progress",
            "seq": [
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
                          [0, 0, 1]]),
                np.array([[0, 0, 0], 
                          [0, -1, 0],
                          [0, 0, 1]]),
                np.array([[0, 0, 0], 
                          [0, -1, 0],
                          [0, 0, 1]]),
            ]
        },
        5: {
            "description": "5 step progress",
            "seq": [
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
                          [0, 1, 0]]),
                np.array([[0, 0, 0], 
                          [0, -1, 0],
                          [0, 1, 0]]),
            ]
        },
        6: {
            "description": "6 step progress",
            "seq": [
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
                          [0, 1, 0]]),
                np.array([[0, 0, 0], 
                          [0, -1, 0],
                          [1, 0, 0]]),
                np.array([[0, 0, 0], 
                          [0, -1, 0],
                          [1, 0, 0]]),
            ]
        },
        11: {
            "description": "1 step reverse progress",
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
                np.array([[0, 0, 0], 
                          [1, -1, 0],
                          [0, 0, 0]]),
                np.array([[0, 0, 0], 
                          [1, -1, 0],
                          [0, 0, 0]]),
            ]
        },
        12: {
            "description": "2 step reverse progress",
            "seq": [
                np.array([[0, 0, 0], 
                          [1, -1, 0],
                          [0, 0, 0]]),
                np.array([[0, 0, 0], 
                          [0, -1, 0],
                          [1, 0, 0]]),
                np.array([[0, 0, 0], 
                          [0, -1, 0],
                          [1, 0, 0]]),
                np.array([[0, 0, 0], 
                          [0, -1, 0],
                          [1, 0, 0]]),
                np.array([[0, 0, 0], 
                          [0, -1, 0],
                          [1, 0, 0]]),
                np.array([[0, 0, 0], 
                          [0, -1, 0],
                          [1, 0, 0]]),
                np.array([[0, 0, 0], 
                          [0, -1, 0],
                          [1, 0, 0]]),
            ]
        },
        13: {
            "description": "3 step reverse progress",
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
        },
        14: {
            "description": "4 step reverse progress",
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
                          [0, -1, 0],
                          [0, 0, 1]]),
                np.array([[0, 0, 0], 
                          [0, -1, 0],
                          [0, 0, 1]]),
                np.array([[0, 0, 0], 
                          [0, -1, 0],
                          [0, 0, 1]]),
            ]
        }
    }
}
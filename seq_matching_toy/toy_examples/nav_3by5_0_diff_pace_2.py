import numpy as np

"""
Making the 2 goals equally far from the start. And the distance between the 2 goals are also equal to the distance between the start and a goal.

When moving in general, the reference can directly jump to the cell that they want to go to.
However, when moving at the last common, the reference has to move only one step at a time.
"""
nav_3by5_0_diff_pace_2 = {
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
        np.array([[1, 0, 0, 0, 0],
                  [0, -1, -1, -1, 0],
                  [0, 0, 0, 0, 0]]),  # Reach the 1st goal
        np.array([[0, 0, 0, 1, 0],
                  [0, -1, -1, -1, 0],
                  [0, 0, 0, 0, 0]]),
        np.array([[0, 0, 0, 0, 0],
                  [0, -1, -1, -1, 0],
                  [0, 0, 0, 1, 0]]),
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
            ]
        },
        
    }
}
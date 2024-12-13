import numpy as np


lava_cycle_enforced_longer = {
    "plot": {
        "reward_vmin": 0,
        "reward_vmax": 10,
    },
    "starting_pos": np.array([1, 1, 0]),
    "map_array": np.array([
        [-1, -1, -1, -1, -1, -1, -1],
        [-1,  0,  0,  0,  0,  0, -1],
        [-1,  1, -1, -1, -1,  0, -1],
        [-1,  0,  0,  0,  0,  0, -1],
        [-1, -1, -1, -1, -1, -1, -1],
    ]),
    "ref_seq": np.array([
                        [3, 1, 0],
                        [4, 1, 0],
                        [4, 3, 2],
                        [3, 3, 1],
                        [1, 3, 0],
                         ]),
    "obs_seqs": {}
}

lava_cycle_enforced = {
    "plot": {
        "reward_vmin": 0,
        "reward_vmax": 10,
    },
    "starting_pos": np.array([1, 1, 1]),
    "map_array": np.array([
        [-1, -1, -1, -1, -1, -1, -1],
        [-1,  0,  0,  0,  0,  0, -1],
        [-1,  1, -1, -1, -1,  0, -1],
        [-1,  0,  0,  0,  0,  0, -1],
        [-1, -1, -1, -1, -1, -1, -1],
    ]),
    "ref_seq": np.array([
                        [4, 3, 2],
                        [3, 3, 1],
                        [1, 3, 0],
                         ]),
    "obs_seqs": {}
}


lava_cycle_reverse = {
    "plot": {
        "reward_vmin": 0,
        "reward_vmax": 10,
    },
    "starting_pos": np.array([1, 1, 1]),
    "map_array": np.array([
        [-1, -1, -1, -1, -1, -1, -1],
        [-1,  0,  0,  0,  0,  0, -1],
        [-1,  1,  1,  1,  1,  0, -1],
        [-1,  0,  0,  0,  0,  0, -1],
        [-1, -1, -1, -1, -1, -1, -1],
    ]),
    "ref_seq": np.array([[5, 1, 3],
                        [5, 3, 3],
                        [3, 3, 0],
                        [1, 3, 1],
                         ]),
    "obs_seqs": {}
}

lava_cycle = {
    "plot": {
        "reward_vmin": 0,
        "reward_vmax": 10,
    },
    "starting_pos": np.array([1, 1, 1]),
    "map_array": np.array([
        [-1, -1, -1, -1, -1, -1, -1],
        [-1,  0,  0,  0,  0,  0, -1],
        [-1,  1,  1,  1,  1,  0, -1],
        [-1,  0,  0,  0,  0,  0, -1],
        [-1, -1, -1, -1, -1, -1, -1],
    ]),
    "ref_seq": np.array([[5, 1, 1],
                        [5, 3, 1],
                        [3, 3, 2],
                        [1, 3, 2],
                         ]),
    "obs_seqs": {
        1: {
            "descriptions": "OT Fail",
            "seq": [
                    np.array([1, 1, 1]),
                    np.array([1, 1, 0]),
                    np.array([2, 1, 0]),
                    np.array([3, 1, 0]),
                    np.array([4, 1, 0]),
                    np.array([4, 1, 1]),
                    np.array([4, 2, 1]),
                    np.array([4, 3, 1]),
                    np.array([4, 3, 1]),
                    np.array([4, 3, 1]),
                    np.array([4, 3, 1]),
                    np.array([4, 3, 2]),
                    np.array([4, 3, 1]),
                    np.array([4, 3, 2]),
                    np.array([3, 3, 2]),
                    np.array([2, 3, 2]),
                    np.array([1, 3, 2]),
                    np.array([1, 3, 2]),
                    np.array([1, 3, 2]),
                    np.array([1, 3, 2]),
                    np.array([1, 3, 2]),
                    np.array([1, 3, 2])
                ]
        },
        2: {
            "descriptions": "DTW Fail",
            "seq": [
                    np.array([1, 1, 1]),
                    np.array([1, 2, 1]),
                    np.array([1, 2, 1]),
                    np.array([1, 3, 1]),
                    np.array([1, 3, 2]),
                    np.array([1, 3, 2]),
                    np.array([1, 3, 2]),
                    np.array([1, 3, 2]),
                    np.array([1, 3, 2]),
                    np.array([1, 3, 2]),
                    np.array([1, 3, 2]),
                    np.array([1, 3, 2]),
                    np.array([1, 3, 2]),
                    np.array([1, 3, 2]),
                    np.array([1, 3, 2]),
                    np.array([1, 3, 2]),
                    np.array([1, 3, 2]),
                    np.array([1, 3, 2]),
                    np.array([1, 3, 2]),
                    np.array([1, 3, 2]),
                ]
        },
        3: {
            "descriptions": "Success",
            "seq": [
                    np.array([1, 1, 1]),
                    np.array([1, 1, 0]),
                    np.array([2, 1, 0]),
                    np.array([3, 1, 0]),
                    np.array([4, 1, 0]),
                    np.array([5, 1, 0]),
                    np.array([5, 1, 1]),
                    np.array([5, 1, 1]),
                    np.array([5, 1, 1]),
                    np.array([5, 2, 1]),
                    np.array([5, 3, 1]),
                    np.array([5, 3, 1]),
                    np.array([5, 3, 1]),
                    np.array([5, 3, 2]),
                    np.array([4, 3, 2]),
                    np.array([3, 3, 2]),
                    np.array([3, 3, 2]),
                    np.array([2, 3, 2]),
                    np.array([1, 3, 2]),
                    np.array([1, 3, 2]),
                ]
        },
        4: {
            "descriptions": "backwards",
            "seq": [
                    np.array([1, 1, 1]),
                    np.array([1, 2, 1]),
                    np.array([1, 3, 1]),
                    np.array([1, 3, 2]),
                    np.array([1, 3, 1]),
                    np.array([1, 3, 0]),
                    np.array([2, 3, 0]),
                    np.array([3, 3, 0]),
                    np.array([3, 3, 1]),
                    np.array([3, 3, 2]),
                    np.array([3, 3, 3]),
                    np.array([3, 3, 0]),
                    np.array([4, 3, 0]),
                    np.array([5, 3, 0]),
                    np.array([5, 3, 1]),
                    np.array([5, 3, 2]),
                    np.array([5, 3, 3]),
                    np.array([5, 2, 3]),
                    np.array([5, 1, 3]),
                    np.array([5, 1, 2]),
                    np.array([5, 1, 1]),

                ]
        }
    }
}

lava_easy = {
    "plot": {
        "reward_vmin": 0,
        "reward_vmax": 10,
    },
    "starting_pos": np.array([1, 1, 0]),
    "map_array": np.array([
        [-1, -1, -1, -1, -1, -1, -1],
        [-1,  0,  0,  1,  0,  0, -1],
        [-1, -1, -1, -1, -1, -1, -1],
    ]),
    "ref_seq": np.array([
                        [5, 1, 0],
                         ]),
    "obs_seqs": {
    0: {
        "descriptions": "Correct",
        "seq": [
            np.array([1, 1, 0]),
            np.array([2, 1, 0]),
            np.array([3, 1, 0]),
            np.array([4, 1, 0]),
            np.array([5, 1, 0]),
        ]
    },
    }
}

lava_periodic = {
    "plot": {
        "reward_vmin": 0,
        "reward_vmax": 10,
    },
    "starting_pos": np.array([1, 1, 0]),
    "map_array": np.array([
        [-1, -1, -1, -1, -1, -1, -1],
        [-1,  0,  0,  0,  0,  0, -1],
        [-1, -1, -1, -1, -1, -1, -1],
    ]),
    "ref_seq": np.array([
                        [2, 1, 0],
                        [4, 1, 0],
                        [5, 1, 0],
                         ]),
    "obs_seqs": {
    0: {
        "descriptions": "Correct",
        "seq": [
            np.array([1, 1, 0]),
            np.array([2, 1, 0]),
            np.array([3, 1, 0]),
            np.array([4, 1, 0]),
            np.array([5, 1, 0]),
        ]
    },
    1: {
        "descriptions": "Correct",
        "seq": [
            np.array([1, 1, 0]),
            np.array([2, 1, 0]),
            np.array([2, 1, 0]),
            np.array([2, 1, 0]),
            np.array([2, 1, 0]),
        ]
    },
    }
}

lava_easy_longer_ref = {
    "plot": {
        "reward_vmin": 0,
        "reward_vmax": 10,
    },
    "starting_pos": np.array([1, 1, 0]),
    "map_array": np.array([
        [-1, -1, -1, -1, -1, -1, -1],
        [-1,  0,  0,  0,  0,  0, -1],
        [-1, -1, -1, -1, -1, -1, -1],
    ]),
    "ref_seq": np.array([
                        [2, 1, 0],
                        [4, 1, 0],
                        [5, 1, 0],
                         ]),
    "obs_seqs": {
    0: {
        "descriptions": "Correct",
        "seq": [
            np.array([1, 1, 0]),
            np.array([2, 1, 0]),
            np.array([3, 1, 0]),
            np.array([4, 1, 0]),
            np.array([5, 1, 0]),
        ]
    },
    1: {
        "descriptions": "Correct",
        "seq": [
            np.array([1, 1, 0]),
            np.array([2, 1, 0]),
            np.array([2, 1, 0]),
            np.array([2, 1, 0]),
            np.array([2, 1, 0]),
        ]
    },
    }
}

lava_nav_bigger =  {
    "plot": {
        "reward_vmin": 0,
        "reward_vmax": 1,
    },
    "starting_pos": np.array([1, 1, 0]),
    "map_array": np.array([
        [-1, -1, -1, -1, -1, -1, -1, -1],
        [-1,  0,  0,  0,  1,  1,  1, -1],
        [-1,  1,  1,  0,  1,  1,  1, -1],
        [-1,  1,  1,  0,  1,  1,  1, -1],
        [-1,  1,  1,  0,  1,  1,  1, -1],
        [-1,  1,  1,  0,  0,  0,  1, -1],
        [-1,  1,  1,  1,  1,  0,  1, -1],
        [-1,  1,  1,  1,  1,  0,  0, -1],
        [-1,  1,  1,  1,  1,  1,  0, -1],
        [-1,  1,  1,  1,  1,  1,  0, -1],
        [-1,  1,  1,  1,  1,  1,  0, -1],
        [-1,  1,  1,  1,  1,  1,  0, -1],
        [-1,  1,  1,  1,  1,  1,  0, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1],
    ]),
    "ref_seq": np.array([
                        [1, 1, 0],
                        [3, 1, 1],
                        [3, 5, 0],
                        [5, 5, 1],
                        [5, 7, 0],
                        [6, 7, 1],
                        [6, 12, 0],
                         ]),
    "obs_seqs": {}
}


lava_nav_no_door = {
    "plot": {
        "reward_vmin": 0,
        "reward_vmax": 1,
    },
    "starting_pos": np.array([1, 1, 0]),
    "map_array": np.array([
        [-1, -1, -1, -1, -1, -1, -1],
        [-1,  0,  0,  0,  0,  0, -1],
        [-1,  1,  1,  0,  1,  1, -1],
        [-1,  0,  0,  0,  0,  0, -1],
        [-1, -1, -1, -1, -1, -1, -1],
    ]),
    "ref_seq": np.array([
                        [3, 1, 1],
                        [3, 2, 1],
                        [3, 3, 0],
                        [5, 3, 0],
                         ]),
    "obs_seqs": {
        0: {
            "descriptions": "Success",
            "seq": [
                    np.array([1, 1, 0]),
                    np.array([2, 1, 0]),
                    np.array([3, 1, 0]),
                    np.array([3, 1, 1]),
                    np.array([3, 1, 1]),
                    np.array([3, 1, 1]),
                    np.array([3, 1, 1]),
                    np.array([3, 1, 1]),
                    np.array([3, 1, 1]),
                    np.array([3, 1, 1]),
                    np.array([3, 2, 1]),
                    np.array([3, 2, 1]),
                    np.array([3, 2, 1]),
                    np.array([3, 3, 1]),
                    np.array([3, 3, 0]),
                    np.array([3, 3, 0]),
                    np.array([3, 3, 0]),
                    np.array([3, 3, 0]),
                    np.array([4, 3, 0]),
                    np.array([5, 3, 0]),
                ]
        },
        1: {
            "descriptions": "Fail",
            "seq": [
                    np.array([1, 1, 0]),
                    np.array([2, 1, 0]),
                    np.array([3, 1, 0]),
                    np.array([3, 1, 1]),
                    np.array([3, 1, 1]),
                    np.array([3, 1, 1]),
                    np.array([3, 1, 1]),
                    np.array([3, 1, 1]),
                    np.array([3, 1, 1]),
                    np.array([3, 1, 1]),
                    np.array([3, 2, 1]),
                    np.array([3, 2, 1]),
                    np.array([3, 2, 1]),
                    np.array([3, 3, 1]),
                    np.array([3, 3, 0]),
                    np.array([3, 3, 0]),
                    np.array([3, 3, 0]),
                    np.array([3, 3, 0]),
                    np.array([3, 3, 0]),
                    np.array([4, 3, 0]),
                ]
        }
    }
}


periodic_easy = {
    "plot": {
        "reward_vmin": 0,
        "reward_vmax": 10,
    },
    "starting_pos": np.array([1, 1, 0]),
    "map_array": np.array([
        [-1, -1, -1, -1, -1, -1, -1],
        [-1,  0,  0,  0,  0,  0, -1],
        [-1, -1, -1, -1, -1, -1, -1],
    ]),
    "ref_seq": np.array([
                        [1, 1, 0],
                        [3, 1, 0],
                        [5, 1, 0],
                        [5, 1, 1],
                        [5, 1, 2],
                        [4, 1, 2],
                        [3, 1, 2],
                        #[1, 1, 2], #
                        [1, 1, 0],
                        [3, 1, 0],
                        [5, 1, 0],
                         ]),
    "obs_seqs": {
        1: {
            "descriptions": "Success",
            "seq": [
                    np.array([1, 1, 0]),
                    np.array([2, 1, 0]),
                    np.array([3, 1, 0]),
                    np.array([4, 1, 0]),
                    np.array([5, 1, 0]),
                    np.array([5, 1, 1]),
                    np.array([5, 1, 2]),
                    np.array([4, 1, 2]),
                    np.array([3, 1, 2]),
                    np.array([2, 1, 2]),
                    np.array([1, 1, 2]),
                    np.array([1, 1, 1]),
                    np.array([1, 1, 0]),
                    np.array([2, 1, 0]),
                    np.array([3, 1, 0]),
                    np.array([4, 1, 0]),
                    np.array([5, 1, 0]),
                    np.array([5, 1, 0]),
                    np.array([5, 1, 0]),
                    np.array([5, 1, 0]),
                ]
        },
        2: {
            "descriptions": "Stuck",
            "seq": [
                    np.array([1, 1, 0]),
                    np.array([2, 1, 0]),
                    np.array([3, 1, 0]),
                    np.array([4, 1, 0]),
                    np.array([5, 1, 0]),
                    np.array([5, 1, 0]),
                    np.array([5, 1, 0]),
                    np.array([5, 1, 0]),
                    np.array([5, 1, 0]),
                    np.array([5, 1, 0]),
                    np.array([5, 1, 0]),
                    np.array([5, 1, 0]),
                    np.array([5, 1, 0]),
                    np.array([5, 1, 0]),
                    np.array([5, 1, 0]),
                    np.array([5, 1, 0]),
                    np.array([5, 1, 0]),
                    np.array([5, 1, 0]),
                    np.array([5, 1, 0]),
                    np.array([5, 1, 0]),
                ]
        }
    }
}


lava_nav = {
    "plot": {
        "reward_vmin": 0,
        "reward_vmax": 10,
    },
    "starting_pos": np.array([1, 1, 0]),
    "map_array": np.array([
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1,  0,  0,  0,  0,  0, -1,  0,  0, -1],
        [-1,  1,  1,  0,  1,  1, -1,  0,  0, -1],
        [-1,  0,  0,  0,  0,  0,  2,  0,  0, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    ]),
    "ref_seq": np.array([
                        [3, 1, 1],
                        [3, 2, 1],
                        [3, 3, 0],
                        [5, 3, 0],
                         ]),
    "obs_seqs": {}
}
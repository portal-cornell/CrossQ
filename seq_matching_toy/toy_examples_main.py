from seq_matching_toy.toy_examples.criss_cross_0 import *
from seq_matching_toy.toy_examples.ref_with_skips_0 import *
from seq_matching_toy.toy_examples.nav_3by3_0 import *
from seq_matching_toy.toy_examples.nav_3by3_0_full_ref import *
from seq_matching_toy.toy_examples.nav_3by3_1 import *
from seq_matching_toy.toy_examples.nav_3by3_2 import *
from seq_matching_toy.toy_examples.nav_3by3_2_remove_first import *
from seq_matching_toy.toy_examples.nav_3by5_0 import *
from seq_matching_toy.toy_examples.nav_3by5_0_skip_one import *
from seq_matching_toy.toy_examples.nav_3by5_0_diff_pace_0 import *
from seq_matching_toy.toy_examples.nav_3by5_0_diff_pace_1 import *
from seq_matching_toy.toy_examples.nav_3by5_0_diff_pace_2 import *
from seq_matching_toy.toy_examples.nav_periodic import *
from seq_matching_toy.toy_examples.nav_backwards import *
from seq_matching_toy.toy_examples.nav_backwards_easier import *
from seq_matching_toy.toy_examples.straight_line import *
from seq_matching_toy.toy_examples.straight_line_long import *
# Figures for workshop paper
from seq_matching_toy.toy_examples.nav_2by2_0_ot_fail import *
from seq_matching_toy.toy_examples.nav_1by4_0_dtw_fail import *
from seq_matching_toy.toy_examples.nav_2by2_1_sdtw_fail import *
from seq_matching_toy.toy_examples.nav_1by4_1_sdtw_fail import *
from seq_matching_toy.toy_examples.lava import *

from numpy.typing import NDArray

examples = {
    "criss_cross_0": criss_cross_0,
    "ref_with_skips_0": ref_with_skips_0,
    "nav_3by3_0": nav_3by3_0,
    "nav_3by3_0_full_ref": nav_3by3_0_full_ref,
    "nav_3by3_1": nav_3by3_1,
    "nav_3by3_2": nav_3by3_2,
    "nav_3by3_2_remove_first": nav_3by3_2_remove_first,
    "nav_3by5_0": nav_3by5_0,
    "nav_3by5_0_skip_one": nav_3by5_0_skip_one,
    "nav_3by5_0_diff_pace_0": nav_3by5_0_diff_pace_0,
    "nav_3by5_0_diff_pace_1": nav_3by5_0_diff_pace_1,
    "nav_3by5_0_diff_pace_2": nav_3by5_0_diff_pace_2,
    "nav_periodic": nav_periodic,
    "nav_backwards": nav_backwards,
    "nav_backwards_easier": nav_backwards_easier,
    "straight_line": straight_line,
    "straight_line_long": straight_line_long,
    # Figures for workshop paper
    "nav_2by2_0_ot_fail": nav_2by2_0_ot_fail,
    "nav_1by4_0_dtw_fail": nav_1by4_0_dtw_fail,
    "nav_1by4_1_sdtw_fail": nav_1by4_1_sdtw_fail,
    "nav_2by2_1_sdtw_fail": nav_2by2_1_sdtw_fail,
    # Minigrid
    "lava_cycle": lava_cycle,
    "lava_nav": lava_nav,
    "lava_nav_no_door": lava_nav_no_door,
    "lava_easy": lava_easy,
    "lava_easy_longer_ref": lava_easy_longer_ref,
    "lava_nav_bigger": lava_nav_bigger,
}

def load_observations_from_examples_dict(example_name: str) -> NDArray:
    """
    Load the example observations from the example dictionary.

    Parameters:
        example_name: str
            - The name of the example

    Returns:
        observations: dict{int: NDArray}
            - A dict of N keys with values of shape (episode_length, D) corresponding to N example rollouts
    """
    return examples[example_name]["obs_seqs"]

def load_map_from_example_dict(example_name: str) -> NDArray:
    """
    Load the map from the example dictionary.

    Parameters:
        example_name: str
            - The name of the example

    Returns:
        map_array: NDArray
            - The map array
    """
    return examples[example_name]["map_array"]

def load_starting_pos_from_example_dict(example_name: str) -> NDArray:
    """
    Load the starting position from the example dictionary.

    Parameters:
        example_name: str
            - The name of the example

    Returns:
        starting_pos: NDArray
            - The starting position of the agent
    """
    return examples[example_name]["starting_pos"]

def load_ref_seq_from_example_dict(example_name: str) -> NDArray:
    """
    Load the reference seq from the example dictionary.

    Parameters:
        example_name: str
            - The name of the example

    Returns:
        ref_seq: NDArray
            - The array of reference sequences
    """
    return examples[example_name]["ref_seq"]

def load_reward_vmin_vmax_from_example_dict(example_name: str) -> NDArray:
    """
    Load the reard vmin and vmax from the example dictionary.

    Parameters:
        example_name: str
            - The name of the example

    Returns:
        reward_vmin: float
            - The minimum value of the reward to receive in the environment
        reward_vmax: float
            - The maximum value of the reward to receive in the environment
    """
    return examples[example_name]["plot"]["reward_vmin"], examples[example_name]["plot"]["reward_vmax"]

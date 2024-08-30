WANDB_DIR = "./"

# Used abosolute path because eval is run in a subdirectory
# TODO: There's probably a better way to do this
SEQ_DICT = {
        "arms_bracket_right_final_only": ["/share/portal/hw575/CrossQ/create_demo/demos/arms_bracket_right_joint-state.npy"],
        "arms_bracket_down_final_only": ["/share/portal/hw575/CrossQ/create_demo/demos/arms_bracket_down_joint-state.npy"],
        "both_arms_out_final_only": ["/share/portal/hw575/CrossQ/create_demo/demos/both-arms-out_joint-state.npy"],
        "both_arms_out_with_intermediate": ["/share/portal/hw575/CrossQ/create_demo/demos/left-arm-out_joint-state.npy", "/share/portal/hw575/CrossQ/create_demo/demos/both-arms-out_joint-state.npy"],
        "both_arms_up_final_only": ["/share/portal/hw575/CrossQ/create_demo/demos/arms_bracket_up_joint-state.npy"],
        "both_arms_up_with_intermediate": ["/share/portal/hw575/CrossQ/create_demo/demos/both-arms-out_joint-state.npy", "/share/portal/hw575/CrossQ/create_demo/demos/arms_bracket_up_joint-state.npy"],
        "arms_up_then_down": ["/share/portal/hw575/CrossQ/create_demo/demos/left-arm-out_joint-state.npy", "/share/portal/hw575/CrossQ/create_demo/demos/both-arms-out_joint-state.npy", "/share/portal/hw575/CrossQ/create_demo/demos/right-arm-out_joint-state.npy"],
    }

REWARDS_TO_ENTRY_IN_SEQ = {
    "arms_bracket_right_goal_only_euclidean": "arms_bracket_right_final_only",
    "arms_bracket_right_basic_r": "arms_bracket_right_final_only",
    
    "arms_bracket_down_goal_only_euclidean": "arms_bracket_down_final_only",
    "arms_bracket_down_basic_r": "arms_bracket_down_final_only",

    "both_arms_out_goal_only_euclidean": "both_arms_out_final_only",
    "both_arms_out_seq_euclidean": "both_arms_out_with_intermediate",
    "both_arms_out_basic_r": "both_arms_out_with_intermediate",

    "both_arms_up_goal_only_euclidean": "both_arms_up_final_only",
    "both_arms_up_seq_euclidean": "both_arms_up_with_intermediate",
    "both_arms_up_basic_r": "both_arms_up_with_intermediate",

    "arms_up_then_down_seq_euclidean": "arms_up_then_down",
    "arms_up_then_down_seq_stage_detector": "arms_up_then_down",
    "arms_up_then_down_seq_avg": "arms_up_then_down",
    "arms_up_then_down_basic_r": "arms_up_then_down",
}

DEMOS_DICT = {
    reward_name: SEQ_DICT[seq_name] for reward_name, seq_name in REWARDS_TO_ENTRY_IN_SEQ.items()
}
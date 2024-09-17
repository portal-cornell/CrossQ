WANDB_DIR = "./"

# Used abosolute path because eval is run in a subdirectory
# TODO: There's probably a better way to do this
SEQ_DICT = {
        "arms_bracket_left_final_only": ["/share/portal/hw575/CrossQ/create_demo/demos/arms_bracket_left_joint-state.npy"],

        "arms_bracket_right_final_only": ["/share/portal/hw575/CrossQ/create_demo/demos/arms_bracket_right_joint-state.npy"],

        "arms_bracket_down_final_only": ["/share/portal/hw575/CrossQ/create_demo/demos/arms_bracket_down_joint-state.npy"],

        "arms_bracket_up_final_only": ["/share/portal/hw575/CrossQ/create_demo/demos/arms_bracket_up_joint-state.npy"],

        "arms_crossed_high_final_only": ["/share/portal/hw575/CrossQ/create_demo/demos/arms_crossed_high_joint-state.npy"],

        "left_arm_out_final_only": ["/share/portal/hw575/CrossQ/create_demo/demos/left-arm-out_joint-state.npy"],

        "right_arm_out_final_only": ["/share/portal/hw575/CrossQ/create_demo/demos/right-arm-out_joint-state.npy"],

        "left_arm_extend_wave_higher_final_only": ["/share/portal/hw575/CrossQ/create_demo/demos/left-arm-extend-wave-higher_joint-state.npy"],

        "left_arm_extend_wave_lower_final_only": ["/share/portal/hw575/CrossQ/create_demo/demos/left-arm-extend-wave-lower_joint-state.npy"],

        "right_arm_extend_wave_higher_final_only": ["/share/portal/hw575/CrossQ/create_demo/demos/left-arm-extend-wave-higher_joint-state.npy"],

        "right_arm_extend_wave_lower_final_only": ["/share/portal/hw575/CrossQ/create_demo/demos/left-arm-extend-wave-lower_joint-state.npy"],

        "both_arms_out_final_only": ["/share/portal/hw575/CrossQ/create_demo/demos/both-arms-out_joint-state.npy"],
        "both_arms_out_with_intermediate": ["/share/portal/hw575/CrossQ/create_demo/demos/left-arm-out_joint-state.npy", "/share/portal/hw575/CrossQ/create_demo/demos/both-arms-out_joint-state.npy"],
        
        "both_arms_up_final_only": ["/share/portal/hw575/CrossQ/create_demo/demos/arms_bracket_up_joint-state.npy"],
        "both_arms_up_with_intermediate": ["/share/portal/hw575/CrossQ/create_demo/demos/both-arms-out_joint-state.npy", "/share/portal/hw575/CrossQ/create_demo/demos/arms_bracket_up_joint-state.npy"],

        "default_but_arms_up_final_only": ["/share/portal/hw575/CrossQ/create_demo/demos/default-but-arms-up_geom-xpos.npy"],
        
        "arms_up_then_down": ["/share/portal/hw575/CrossQ/create_demo/demos/left-arm-out_joint-state.npy", "/share/portal/hw575/CrossQ/create_demo/demos/both-arms-out_joint-state.npy", "/share/portal/hw575/CrossQ/create_demo/demos/right-arm-out_joint-state.npy"],
    }

REWARDS_TO_ENTRY_IN_SEQ = {
    "arms_bracket_left_goal_only_euclidean_geom_xpos": "arms_bracket_left_final_only",
    "arms_bracket_left_basic_r": "arms_bracket_left_final_only",

    "arms_bracket_right_goal_only_euclidean": "arms_bracket_right_final_only",
    "arms_bracket_right_goal_only_euclidean_geom_xpos": "arms_bracket_right_final_only",
    "arms_bracket_right_basic_r": "arms_bracket_right_final_only",

    "arms_bracket_down_goal_only_euclidean": "arms_bracket_down_final_only",
    "arms_bracket_down_goal_only_euclidean_geom_xpos": "arms_bracket_down_final_only",
    "arms_bracket_down_basic_r": "arms_bracket_down_final_only",

    "arms_bracket_up_goal_only_euclidean_geom_xpos": "arms_bracket_up_final_only",
    "arms_bracket_up_basic_r": "arms_bracket_up_final_only",

    "arms_crossed_high_goal_only_euclidean_geom_xpos": "arms_crossed_high_final_only",
    "arms_crossed_high_basic_r": "arms_crossed_high_final_only",

    "left_arm_out_goal_only_euclidean": "left_arm_out_final_only",
    "left_arm_out_goal_only_euclidean_geom_xpos": "left_arm_out_final_only",
    "left_arm_out_basic_r": "left_arm_out_final_only",
    "left_arm_out_basic_r_geom_xpos": "left_arm_out_final_only",

    "right_arm_out_goal_only_euclidean": "right_arm_out_final_only",
    "right_arm_out_goal_only_euclidean_geom_xpos": "right_arm_out_final_only",
    "right_arm_out_basic_r": "right_arm_out_final_only",

    "left_arm_extend_wave_higher_goal_only_euclidean": "left_arm_extend_wave_higher_final_only",
    "left_arm_extend_wave_higher_goal_only_euclidean_geom_xpos": "left_arm_extend_wave_higher_final_only",
    "left_arm_extend_wave_higher_basic_r": "left_arm_extend_wave_higher_final_only",

    "left_arm_extend_wave_lower_goal_only_euclidean": "left_arm_extend_wave_lower_final_only",
    "left_arm_extend_wave_lower_goal_only_euclidean_geom_xpos": "left_arm_extend_wave_lower_final_only",
    "left_arm_extend_wave_lower_basic_r": "left_arm_extend_wave_lower_final_only",

    "right_arm_extend_wave_higher_goal_only_euclidean": "right_arm_extend_wave_higher_final_only",
    "right_arm_extend_wave_higher_goal_only_euclidean_geom_xpos": "right_arm_extend_wave_higher_final_only",
    "right_arm_extend_wave_higher_basic_r": "right_arm_extend_wave_higher_final_only",

    "right_arm_extend_wave_lower_goal_only_euclidean": "right_arm_extend_wave_lower_final_only",
    "right_arm_extend_wave_lower_goal_only_euclidean_geom_xpos": "right_arm_extend_wave_lower_final_only",
    "right_arm_extend_wave_lower_basic_r": "right_arm_extend_wave_lower_final_only",

    "both_arms_out_goal_only_euclidean": "both_arms_out_final_only",
    "both_arms_out_goal_only_euclidean_geom_xpos": "both_arms_out_final_only",
    "both_arms_out_seq_euclidean": "both_arms_out_with_intermediate",
    "both_arms_out_basic_r": "both_arms_out_with_intermediate",
    "both_arms_out_basic_r_geom_xpos": "both_arms_out_with_intermediate",

    "both_arms_up_goal_only_euclidean": "both_arms_up_final_only",
    "both_arms_up_goal_only_euclidean_geom_xpos": "both_arms_up_final_only",
    "both_arms_up_seq_euclidean": "both_arms_up_with_intermediate",
    "both_arms_up_basic_r": "both_arms_up_with_intermediate",
    "both_arms_up_basic_r_geom_xpos": "both_arms_up_with_intermediate",

    "default_but_arms_up_goal_only_euclidean": "default_but_arms_up_final_only",
    "default_but_arms_up_goal_only_euclidean_geom_xpos": "default_but_arms_up_final_only",
    "default_but_arms_up_basic_r": "default_but_arms_up_final_only",
    
    "arms_up_then_down_seq_euclidean": "arms_up_then_down",
    "arms_up_then_down_seq_stage_detector": "arms_up_then_down",
    "arms_up_then_down_seq_avg": "arms_up_then_down",
    "arms_up_then_down_basic_r": "arms_up_then_down",
    "arms_up_then_down_basic_r_geom_xpos": "arms_up_then_down",
}
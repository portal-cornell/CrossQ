WANDB_DIR = "./"

# Used abosolute path because eval is run in a subdirectory
# TODO: There's probably a better way to do this
TASK_SEQ_DICT = {
    ################################## Sequence Following Tasks
    ########### 2 Key Poses to Follow (2 Key Frames)
    ########### 3 Key Poses to Follow (3 Key Frames)
    "arms_up_then_down":
    {
        "task_type": "sequence_following",
        "sequences": {
            "key_frames": ["/share/portal/hw575/CrossQ/create_demo/demos/left-arm-out_geom-xpos.npy", "/share/portal/hw575/CrossQ/create_demo/demos/both-arms-out_geom-xpos.npy", "/share/portal/hw575/CrossQ/create_demo/demos/right-arm-out_geom-xpos.npy"]
        }
    },
    ################################## Goal Reaching Tasks
    # TODO: For now, we are only using the following tasks
    "left_arm_out":
    {   
        "task_type": "goal_reaching",
        "sequences": {
            "key_frames": ["/share/portal/hw575/CrossQ/create_demo/demos/left-arm-out_geom-xpos.npy"],
            "intermediate_5_frames": "/share/portal/hw575/CrossQ/create_demo/seq_demos/left-arm-out_5-frames_geom-xpos.npy",
        }
    },
    "right_arm_out":
    {
        "task_type": "goal_reaching",
        "sequences": {
            "key_frames": ["/share/portal/hw575/CrossQ/create_demo/demos/right-arm-out_geom-xpos.npy"],
        }
    },
    "left_arm_extend_wave_higher":
    {
        "task_type": "goal_reaching",
        "sequences": {
            "key_frames": ["/share/portal/hw575/CrossQ/create_demo/demos/left-arm-extend-wave-higher_geom-xpos.npy"],
        }
    },
    "right_arm_extend_wave_higher":
    {
        "task_type": "goal_reaching",
        "sequences": {
            "key_frames": ["/share/portal/hw575/CrossQ/create_demo/demos/right-arm-extend-wave-higher_geom-xpos.npy"],
            # Using interpolated sequences as reference sequences
            "intermediate_3_frames": "/share/portal/hw575/CrossQ/create_demo/seq_demos/right-arm-extend-wave-higher_3-frames_geom-xpos.npy",
            "intermediate_5_frames": "/share/portal/hw575/CrossQ/create_demo/seq_demos/right-arm-extend-wave-higher_5-frames_geom-xpos.npy",
            "intermediate_10_frames": "/share/portal/hw575/CrossQ/create_demo/seq_demos/right-arm-extend-wave-higher_10-frames_geom-xpos.npy",
            "intermediate_20_frames": "/share/portal/hw575/CrossQ/create_demo/seq_demos/right-arm-extend-wave-higher_20-frames_geom-xpos.npy",
            "intermediate_30_frames": "/share/portal/hw575/CrossQ/create_demo/seq_demos/right-arm-extend-wave-higher_30-frames_geom-xpos.npy",
            "intermediate_40_frames": "/share/portal/hw575/CrossQ/create_demo/seq_demos/right-arm-extend-wave-higher_40-frames_geom-xpos.npy",
            "intermediate_50_frames": "/share/portal/hw575/CrossQ/create_demo/seq_demos/right-arm-extend-wave-higher_50-frames_geom-xpos.npy",
            "intermediate_60_frames": "/share/portal/hw575/CrossQ/create_demo/seq_demos/right-arm-extend-wave-higher_60-frames_geom-xpos.npy",
            # Using interpolated sequences as reference sequences (which are the last N frames of interpolation)
            "intermediate_last_10_frames": "/share/portal/hw575/CrossQ/create_demo/seq_demos/right-arm-extend-wave-higher_last-10-frames_geom-xpos.npy",
            "intermediate_last_20_frames": "/share/portal/hw575/CrossQ/create_demo/seq_demos/right-arm-extend-wave-higher_last-20-frames_geom-xpos.npy",
            "intermediate_last_30_frames": "/share/portal/hw575/CrossQ/create_demo/seq_demos/right-arm-extend-wave-higher_last-30-frames_geom-xpos.npy",
            "intermediate_last_40_frames": "/share/portal/hw575/CrossQ/create_demo/seq_demos/right-arm-extend-wave-higher_last-40-frames_geom-xpos.npy",
            "intermediate_last_50_frames": "/share/portal/hw575/CrossQ/create_demo/seq_demos/right-arm-extend-wave-higher_last-50-frames_geom-xpos.npy",
            "intermediate_last_60_frames": "/share/portal/hw575/CrossQ/create_demo/seq_demos/right-arm-extend-wave-higher_last-60-frames_geom-xpos.npy",
            # Using actual rollouts as reference sequences
            "rollout_9_frames": "/share/portal/hw575/CrossQ/create_demo/seq_demos/right-arm-extend-wave-higher_real-rollout_9-frames_geom-xpos.npy",
            "rollout_19_frames": "/share/portal/hw575/CrossQ/create_demo/seq_demos/right-arm-extend-wave-higher_real-rollout_19-frames_geom-xpos.npy",
            'rollout_29_frames': "/share/portal/hw575/CrossQ/create_demo/seq_demos/right-arm-extend-wave-higher_real-rollout_29-frames_geom-xpos.npy",
            'rollout_39_frames': "/share/portal/hw575/CrossQ/create_demo/seq_demos/right-arm-extend-wave-higher_real-rollout_39-frames_geom-xpos.npy",
            'rollout_49_frames': "/share/portal/hw575/CrossQ/create_demo/seq_demos/right-arm-extend-wave-higher_real-rollout_49-frames_geom-xpos.npy",
            "rollout_59_frames": "/share/portal/hw575/CrossQ/create_demo/seq_demos/right-arm-extend-wave-higher_real-rollout_59-frames_geom-xpos.npy",
            "handpick_1_frames": "/share/portal/hw575/CrossQ/create_demo/seq_demos/right-arm-extend-wave-higher_hand-picked-rollout_1-frames_geom-xpos.npy",
            "handpick_1_frames_interpolation": "/share/portal/hw575/CrossQ/create_demo/seq_demos/right-arm-extend-wave-higher_hand-picked-rollout_1-frames_from-interpolation_geom-xpos.npy",
            "handpick_2_frames": "/share/portal/hw575/CrossQ/create_demo/seq_demos/right-arm-extend-wave-higher_hand-picked-rollout_2-frames_n=13-111_geom-xpos.npy",
            "handpick_2_frames_106-111": "/share/portal/hw575/CrossQ/create_demo/seq_demos/right-arm-extend-wave-higher_hand-picked-rollout_2-frames_n=106-111_geom-xpos.npy",
            "handpick_3_frames": "/share/portal/hw575/CrossQ/create_demo/seq_demos/right-arm-extend-wave-higher_hand-picked-rollout_3-frames_geom-xpos.npy",
        }
    },
    "both_arms_out":
    {
        "task_type": "goal_reaching",
        "sequences": {
            "key_frames": ["/share/portal/hw575/CrossQ/create_demo/demos/both-arms-out_geom-xpos.npy"],
        }
    },
    # All the tasks (but some not really achievable)
    # "arms_bracket_left":
    # {
    #     "task_type": "goal_reaching",
    #     "sequences": {
    #         "key_frames": ["/share/portal/hw575/CrossQ/create_demo/demos/arms_bracket_left_geom-xpos.npy"],
    #     }
    # },
    # "arms_bracket_right":
    # {
    #     "task_type": "goal_reaching",
    #     "sequences": {
    #         "key_frames": ["/share/portal/hw575/CrossQ/create_demo/demos/arms_bracket_right_geom-xpos.npy"],
    #     }
    # },
    # "arms_bracket_down":
    # {
    #     "task_type": "goal_reaching",
    #     "sequences": {
    #         "key_frames": ["/share/portal/hw575/CrossQ/create_demo/demos/arms_bracket_down_geom-xpos.npy"],
    #     }
    # },
    # "arms_bracket_up":
    # {
    #     "task_type": "goal_reaching",
    #     "sequences": {
    #         "key_frames": ["/share/portal/hw575/CrossQ/create_demo/demos/arms_bracket_up_geom-xpos.npy"],
    #     }
    # },
    # "arms_crossed_high":
    # {
    #     "task_type": "goal_reaching",
    #     "sequences": {
    #         "key_frames": ["/share/portal/hw575/CrossQ/create_demo/demos/arms_crossed_high_geom-xpos.npy"],
    #     }
    # },
    # "left_arm_extend_wave_higher":
    # {
    #     "task_type": "goal_reaching",
    #     "sequences": {
    #         "key_frames": ["/share/portal/hw575/CrossQ/create_demo/demos/left-arm-extend-wave-higher_geom-xpos.npy"],
    #     }
    # },
    # "left_arm_extend_wave_lower":
    # {
    #     "task_type": "goal_reaching",
    #     "sequences": {
    #         "key_frames": ["/share/portal/hw575/CrossQ/create_demo/demos/left-arm-extend-wave-lower_geom-xpos.npy"],
    #     }
    # },
    # "right_arm_extend_wave_higher":
    # {
    #     "task_type": "goal_reaching",
    #     "sequences": {
    #         "key_frames": ["/share/portal/hw575/CrossQ/create_demo/demos/right-arm-extend-wave-higher_geom-xpos.npy"],
    #     }
    # },
    # "right_arm_extend_wave_lower":
    # {
    #     "task_type": "goal_reaching",
    #     "sequences": {
    #         "key_frames": ["/share/portal/hw575/CrossQ/create_demo/demos/right-arm-extend-wave-lower_geom-xpos.npy"],
    #     }
    # },
    # "both_arms_out":
    # {
    #     "task_type": "goal_reaching",
    #     "sequences": {
    #         "key_frames": ["/share/portal/hw575/CrossQ/create_demo/demos/both-arms-out_geom-xpos.npy"],
    #     }
    # },
}

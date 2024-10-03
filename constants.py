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
            "intermediate_5_frames": "/share/portal/hw575/CrossQ/create_demo/seq_demos/right-arm-extend-wave-higher_5-frames_geom-xpos.npy",
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

WANDB_DIR = "./"

METAWORLD_TASK_SEQ_DICT = {
    # ==================== Easy Tasks ====================
    "button-press-v2":
    {
        "num_frames": 125, #20,
        "last_frame": 125, #90,
        "task_type": "sequence_following",
        "sequences": {
            "rl_expert": "/share/portal/hw575/CrossQ/train_logs/2024-11-25-125324_sb3_sac_envt=button-press-v2-goal-hidden_rm=hand_engineered_nt=ep-len=200_sparse/eval/1000000_rollouts_states.npy",
            "rl_expert_20_frames": "/share/portal/wph52/CrossQ/ref_seqs/button_press/1000000_rollouts_subsampled_20_states.npy",
            "hand_engineered_corner": "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/button-press-v2/button-press-v2_corner_0_states.npy"
        },
        "sequence_gifs": {
            "rl_expert": "/share/portal/hw575/CrossQ/train_logs/2024-11-25-125324_sb3_sac_envt=button-press-v2-goal-hidden_rm=hand_engineered_nt=ep-len=200_sparse/eval/1000000_rollouts.gif",
            "rl_expert_20_frames": "/share/portal/wph52/CrossQ/ref_seqs/button_press/1000000_rollouts_subsampled_20.gif",
            "hand_engineered_corner": "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/button-press-v2/button-press-v2_corner_0.gif"
        }
    },
    "door-close-v2":
    {
        "num_frames": 125, #20,
        "last_frame": 125, #76,
        "task_type": "sequence_following",
        "sequences": {
            "rl_expert": "/share/portal/hw575/CrossQ/train_logs/2024-11-25-124432_sb3_sac_envt=door-close-v2-goal-hidden_rm=hand_engineered_nt=ep-len=200_sparse/eval/1000000_rollouts_states.npy",
            "rl_expert_20_frames": "/share/portal/wph52/CrossQ/ref_seqs/door_close/1000000_rollouts_subsampled_20_states.npy",
            "hand_engineered_corner": "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/door-close-v2/door-close-v2_corner_0_states.npy"
        },
        "sequence_gifs": {
            "rl_expert": "/share/portal/hw575/CrossQ/train_logs/2024-11-25-124432_sb3_sac_envt=door-close-v2-goal-hidden_rm=hand_engineered_nt=ep-len=200_sparse/eval/1000000_rollouts.gif",
            "rl_expert_20_frames": "/share/portal/wph52/CrossQ/ref_seqs/door_close/1000000_rollouts_subsampled_20.gif",
            "hand_engineered_corner": "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/door-close-v2/door-close-v2_corner_0.gif"
        }
    },
    # ==================== Medium Tasks ====================
    "door-open-v2":
    {
        "num_frames": 20,
        "last_frame": 100,
        "task_type": "sequence_following",
        "sequences": {
            "rl_expert": "/share/portal/hw575/CrossQ/train_logs/2024-12-09-212051_sb3_sac_envt=door-open-v2-goal-observable_rm=hand_engineered_nt=dense/eval/1000000_rollouts_states.npy",
            "rl_expert_corner3": "/share/portal/hw575/CrossQ/train_logs/2024-12-15-183942_sb3_sac_envt=door-open-v2-goal-observable_rm=hand_engineered_nt=dense_corner3/eval/1000000_rollouts_states.npy",
            "hand_engineered_corner3": "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/door-open-v2/door-open-v2_corner3_0_states.npy"
        },
        "sequence_gifs": {
            "rl_expert": "/share/portal/hw575/CrossQ/train_logs/2024-12-09-212051_sb3_sac_envt=door-open-v2-goal-observable_rm=hand_engineered_nt=dense/eval/1000000_rollouts.gif",
            "rl_expert_corner3": "/share/portal/hw575/CrossQ/train_logs/2024-12-15-183942_sb3_sac_envt=door-open-v2-goal-observable_rm=hand_engineered_nt=dense_corner3/eval/1000000_rollouts.gif",
            "hand_engineered_corner3": "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/door-open-v2/door-open-v2_corner3_0.gif"
        }
    },
    "window-open-v2":
    {
        "num_frames": 20,
        "last_frame": 94,
        "task_type": "sequence_following",
        "sequences": {
            "rl_expert": "/share/portal/hw575/CrossQ/train_logs/2024-12-09-212409_sb3_sac_envt=window-open-v2-goal-observable_rm=hand_engineered_nt=dense/eval/940000_rollouts_states.npy",
            "rl_expert_corner3": "/share/portal/hw575/CrossQ/train_logs/2024-12-15-181003_sb3_sac_envt=window-open-v2-goal-observable_rm=hand_engineered_nt=dense_corner3/eval/1000000_rollouts_states.npy",
            "hand_engineered_corner3": "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/window-open-v2/window-open-v2_corner3_0_states.npy"
        },
        "sequence_gifs": {
            "rl_expert": "/share/portal/hw575/CrossQ/train_logs/2024-12-09-212409_sb3_sac_envt=window-open-v2-goal-observable_rm=hand_engineered_nt=dense/eval/940000_rollouts.gif",
            "rl_expert_corner3": "/share/portal/hw575/CrossQ/train_logs/2024-12-15-181003_sb3_sac_envt=window-open-v2-goal-observable_rm=hand_engineered_nt=dense_corner3/eval/1000000_rollouts.gif",
            "hand_engineered_corner3": "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/window-open-v2/window-open-v2_corner3_0.gif"
        }
    },
    "lever-pull-v2":
    {
        
        "num_frames": 20,
        "last_frame": 111, 
        "task_type": "sequence_following",
        "sequences": {
            # best view
            "hand_engineered_corner2": "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/lever-pull-v2/lever-pull-v2_corner2_0_states.npy",
            # temporalOT
            "hand_engineered_corner4": "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/lever-pull-v2/lever-pull-v2_corner4_0_states.npy"
        },
        "sequence_gifs": {
            "hand_engineered_corner2": "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/lever-pull-v2/lever-pull-v2_corner2_0.gif",
            "hand_engineered_corner4": "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/lever-pull-v2/lever-pull-v2_corner4_0.gif"
        }
    },
    "hand-insert-v2":
    {
        "num_frames": 20,
        "last_frame": 78, 
        "task_type": "sequence_following",
        "sequences": {
            # best view
            "hand_engineered_corner3": "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/hand-insert-v2/hand-insert-v2_corner_0_states.npy",
            # temporalOT
            "hand_engineered_corner": "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/hand-insert-v2/hand-insert-v2_corner_0_states.npy"
        },
        "sequence_gifs": {
            "hand_engineered_corner3": "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/hand-insert-v2/hand-insert-v2_corner_0.gif",
            "hand_engineered_corner": "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/hand-insert-v2/hand-insert-v2_corner_0.gif"
        }
    },
    "push-v2": {
        "num_frames": 20,
        "last_frame": 93,
        "task_type": "sequence_following",
        "sequences": {
            "hand_engineered_corner3": "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/push-v2/push-v2_corner3_0_states.npy"
        },
        "sequence_gifs": {
            "hand_engineered_corner3": "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/push-v2/push-v2_corner3_0.gif"
        }
    },
    "basketball-v2": {
        "num_frames": 20,
        "last_frame": 110,
        "task_type": "sequence_following",
        "sequences": {
            "hand_engineered_corner": "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/basketball-v2/basketball-v2_corner_0_states.npy"
        },
        "sequence_gifs": {
            "hand_engineered_corner": "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/basketball-v2/basketball-v2_corner_0.gif"
        }
    },
    "stick-push-v2": {
        "num_frames": 20,
        "last_frame": 92,
        "task_type": "sequence_following",
        "sequences": {
            # best view
            "hand_engineered_corner4": "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/stick-push-v2/stick-push-v2_corner4_0_states.npy",
            # temporalOT
            "hand_engineered_corner": "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/stick-push-v2/stick-push-v2_corner_0_states.npy"
        },
        "sequence_gifs": {
            "hand_engineered_corner4": "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/stick-push-v2/stick-push-v2_corner4_0.gif",
            "hand_engineered_corner": "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/stick-push-v2/stick-push-v2_corner_0.gif"
        }
    },
    "door-lock-v2": {
        "num_frames": 125, #20,
        "last_frame": 125, #85,
        "task_type": "sequence_following",
        "sequences": {
            "hand_engineered_corner": "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/door-lock-v2/door-lock-v2_corner_0_states.npy"
        },
        "sequence_gifs": {
            "hand_engineered_corner": "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/door-lock-v2/door-lock-v2_corner_0.gif"
        }
    },
    "bin-picking-v2": {
        "num_frames": 20,
        "last_frame": 114,
        "task_type": "sequence_following",
        "sequences": {
            "hand_engineered_corner": "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/bin-picking-v2/bin-picking-v2_corner_0_states.npy"
        },
        "sequence_gifs": {
            "hand_engineered_corner": "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/bin-picking-v2/bin-picking-v2_corner_0.gif"
        }
    },
    "box-close-v2": {
        "num_frames": 175, # 20,
        "last_frame": 175, # 109,
        "task_type": "sequence_following",
        "sequences": {
            "hand_engineered_corner3": "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/box-close-v2/box-close-v2_corner3_0_states.npy"
        },
        "sequence_gifs": {
            "hand_engineered_corner3": "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/box-close-v2/box-close-v2_corner3_0.gif"
        }
    },
    "pick-place-v2": {
        "num_frames": 20,
        "last_frame": 75,
        "task_type": "sequence_following",
        "sequences": {
            "hand_engineered_corner3": "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/pick-place-v2/pick-place-v2_corner3_0_states.npy"
        },
        "sequence_gifs": {
            "hand_engineered_corner3": "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/pick-place-v2/pick-place-v2_corner3_0.gif"
        }
    },
    "assembly-v2": {
        "num_frames": 175, #20,
        "last_frame": 175, #110,
        "task_type": "sequence_following",
        "sequences": {
            "hand_engineered_corner": "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/assembly-v2/assembly-v2_corner_0_states.npy"
        },
        "sequence_gifs": {
            "hand_engineered_corner": "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/assembly-v2/assembly-v2_corner_0.gif"
        }
    },
    "disassemble-v2": {
        "num_frames": 20,
        "last_frame": 108,
        "task_type": "sequence_following",
        "sequences": {
            "hand_engineered_corner": "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/disassemble-v2/disassemble-v2_corner_0_states.npy"
        },
        "sequence_gifs": {
            "hand_engineered_corner": "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/disassemble-v2/disassemble-v2_corner_0.gif"
        }
    },
    "hammer-v2": {
        "num_frames": 125, # 20,
        "last_frame": 125, #81,
        "task_type": "sequence_following",
        "sequences": {
            "hand_engineered_corner3": "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/hammer-v2/hammer-v2_corner3_0_states.npy"
        },
        "sequence_gifs": {
            "hand_engineered_corner3": "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/hammer-v2/hammer-v2_corner3_0.gif"
        }
    },
    "peg-insert-side-v2": {
        "num_frames": 20,
        "last_frame": 102,
        "task_type": "sequence_following",
        "sequences": {
            "hand_engineered_corner3": "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/peg-insert-side-v2/peg-insert-side-v2_corner3_0_states.npy"
        },
        "sequence_gifs": {
            "hand_engineered_corner3": "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/peg-insert-side-v2/peg-insert-side-v2_corner3_0.gif"
        }
    },
    "door-unlock-v2": {
        "num_frames": 20,
        "last_frame": 59,
        "task_type": "sequence_following",
        "sequences": {
            "hand_engineered_corner": "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/door-unlock-v2/door-unlock-v2_corner_0_states.npy"
        },
        "sequence_gifs": {
            "hand_engineered_corner": "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/door-unlock-v2/door-unlock-v2_corner_0.gif"
        }
    }

}

# Used abosolute path because eval is run in a subdirectory
# TODO: There's probably a better way to do this
HUMANOID_TASK_SEQ_DICT = {
    ################################## Sequence Following Tasks
    ########### 2 Key Poses to Follow (2 Key Frames)
    "right_arm_out_to_both_arms_out": 
    {
        "task_type": "sequence_following",
        "sequences": {
            "key_frames": ["/share/portal/hw575/CrossQ/create_demo/demos/right-arm-out_geom-xpos.npy", "/share/portal/hw575/CrossQ/create_demo/demos/both-arms-out_geom-xpos.npy"],
            # Using interpolated sequences as reference sequences
            #   The name indicate the total number of frames
            #   So each key pose gets N / 2 frames
            "intermediate_10_frames": "/share/portal/hw575/CrossQ/create_demo/seq_demos/right-arm-out_to_both-arms-out_10-frames_geom-xpos.npy",
            "intermediate_20_frames": "/share/portal/hw575/CrossQ/create_demo/seq_demos/right-arm-out_to_both-arms-out_20-frames_geom-xpos.npy",
        }
    },
    "left_arm_out_to_left_arm_extend_wave_higher":
    {
        "task_type": "sequence_following",
        "sequences": {
            "key_frames": ["/share/portal/hw575/CrossQ/create_demo/demos/left-arm-out_geom-xpos.npy", "/share/portal/hw575/CrossQ/create_demo/demos/left-arm-extend-wave-higher_geom-xpos.npy"],
            # Using interpolated sequences as reference sequences
            "intermediate_10_frames": "/share/portal/hw575/CrossQ/create_demo/seq_demos/left-arm-out_to_left-arm-extend-wave-higher_10-frames_geom-xpos.npy",
            "intermediate_20_frames": "/share/portal/hw575/CrossQ/create_demo/seq_demos/left-arm-out_to_left-arm-extend-wave-higher_20-frames_geom-xpos.npy",
        }
    },
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
            # Using interpolated sequences as reference sequences
            "intermediate_10_frames": "/share/portal/hw575/CrossQ/create_demo/seq_demos/left-arm-out_10-frames_geom-xpos.npy",
            "intermediate_20_frames": "/share/portal/hw575/CrossQ/create_demo/seq_demos/left-arm-out_20-frames_geom-xpos.npy",
            "intermediate_30_frames": "/share/portal/hw575/CrossQ/create_demo/seq_demos/left-arm-out_30-frames_geom-xpos.npy",
            "intermediate_40_frames": "/share/portal/hw575/CrossQ/create_demo/seq_demos/left-arm-out_40-frames_geom-xpos.npy",
        }
    },
    "right_arm_out":
    {
        "task_type": "goal_reaching",
        "sequences": {
            "key_frames": ["/share/portal/hw575/CrossQ/create_demo/demos/right-arm-out_geom-xpos.npy"],
            # Using interpolated sequences as reference sequences
            "intermediate_10_frames": "/share/portal/hw575/CrossQ/create_demo/seq_demos/right-arm-out_10-frames_geom-xpos.npy",
            "intermediate_20_frames": "/share/portal/hw575/CrossQ/create_demo/seq_demos/right-arm-out_20-frames_geom-xpos.npy",
            "intermediate_30_frames": "/share/portal/hw575/CrossQ/create_demo/seq_demos/right-arm-out_30-frames_geom-xpos.npy",
            "intermediate_40_frames": "/share/portal/hw575/CrossQ/create_demo/seq_demos/right-arm-out_40-frames_geom-xpos.npy",
        }
    },
    "left_arm_extend_wave_higher":
    {
        "task_type": "goal_reaching",
        "sequences": {
            "key_frames": ["/share/portal/hw575/CrossQ/create_demo/demos/left-arm-extend-wave-higher_geom-xpos.npy"],
            # Using interpolated sequences as reference sequences
            "intermediate_10_frames": "/share/portal/hw575/CrossQ/create_demo/seq_demos/left-arm-extend-wave-higher_10-frames_geom-xpos.npy",
            "intermediate_20_frames": "/share/portal/hw575/CrossQ/create_demo/seq_demos/left-arm-extend-wave-higher_20-frames_geom-xpos.npy",
            "intermediate_30_frames": "/share/portal/hw575/CrossQ/create_demo/seq_demos/left-arm-extend-wave-higher_30-frames_geom-xpos.npy",
            "intermediate_40_frames": "/share/portal/hw575/CrossQ/create_demo/seq_demos/left-arm-extend-wave-higher_40-frames_geom-xpos.npy",
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
            # Using interpolated sequences as reference sequences
            "intermediate_10_frames": "/share/portal/hw575/CrossQ/create_demo/seq_demos/both-arms-out_10-frames_geom-xpos.npy",
            "intermediate_20_frames": "/share/portal/hw575/CrossQ/create_demo/seq_demos/both-arms-out_20-frames_geom-xpos.npy",
            "intermediate_30_frames": "/share/portal/hw575/CrossQ/create_demo/seq_demos/both-arms-out_30-frames_geom-xpos.npy",
            "intermediate_40_frames": "/share/portal/hw575/CrossQ/create_demo/seq_demos/both-arms-out_40-frames_geom-xpos.npy",
        }
    },
    "both_arms_down":
    {
        "task_type": "goal_reaching",
        "sequences": {
            "key_frames": ["/share/portal/hw575/CrossQ/create_demo/demos/both-arms-down_geom-xpos.npy"],
            # Using interpolated sequences as reference sequences
            "intermediate_10_frames": "/share/portal/hw575/CrossQ/create_demo/seq_demos/both-arms-down_10-frames_geom-xpos.npy",
            "intermediate_20_frames": "/share/portal/hw575/CrossQ/create_demo/seq_demos/both-arms-down_20-frames_geom-xpos.npy",
            "intermediate_30_frames": "/share/portal/hw575/CrossQ/create_demo/seq_demos/both-arms-down_30-frames_geom-xpos.npy",
            "intermediate_40_frames": "/share/portal/hw575/CrossQ/create_demo/seq_demos/both-arms-down_40-frames_geom-xpos.npy",
        }
    }
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


METAWORLD_EPISODE_LENGTH = {
    'door-close-v2': 125, # Added by us
    'button-press-v2': 125, # Added by us
    'hammer-v2': 125,
    'drawer-close-v2': 125,
    'drawer-open-v2': 125,
    'door-open-v2': 125,
    'bin-picking-v2': 175,
    'button-press-topdown-v2': 125,
    'door-unlock-v2': 125,
    'basketball-v2': 175,
    'plate-slide-v2': 125,
    'hand-insert-v2': 125,
    'peg-insert-side-v2': 150,
    'soccer-v2': 125,
    'assembly-v2': 175,
    'disassemble-v2': 125,
    'pick-place-wall-v3': 175,
    'pick-place-v2': 125,
    'push-v2': 125,
    'push-wall-v2': 175,
    'lever-pull-v2': 175,
    'stick-pull-v2': 175,
    'shelf-place-v2': 175,
    'window-close-v2': 125,
    'window-open-v2': 125,
    'reach-v2': 125,
    'button-press-wall-v2': 125,
    'box-close-v2': 175,
    'stick-push-v2': 125,
    'handle-pull-v2': 175,
    'door-lock-v2': 125,
}


METAWORLD_DEFAULT_CAMERA = {
    'button-press-v2': 'corner', # Added by us
    'door-close-v2': 'corner', # Added by us
    'hammer-v2': 'corner3',
    'drawer-close-v2': 'corner',
    'drawer-open-v2': 'corner',
    'door-open-v2': 'corner3',
    'bin-picking-v2': 'corner',
    'button-press-topdown-v2': 'corner',
    'door-unlock-v2': 'corner',
    'basketball-v2': 'corner',
    'plate-slide-v2': 'corner',
    'hand-insert-v2': 'corner',
    'peg-insert-side-v2': 'corner3',
    'soccer-v2': 'corner',
    'assembly-v2': 'corner',
    'disassemble-v2': 'corner',
    'pick-place-wall-v3': 'corner3',
    'pick-place-v2': 'corner3',
    'push-v2': 'corner3',
    'push-wall-v2': 'corner',
    'lever-pull-v2': 'corner4',
    'stick-pull-v2': 'corner3',
    'shelf-place-v2': 'corner',
    'window-close-v2': 'corner3',
    'window-open-v2': 'corner3',
    'reach-v2': 'corner3',
    'button-press-wall-v2': 'corner',
    'box-close-v2': 'corner3',
    'stick-push-v2': 'corner',
    'handle-pull-v2': 'corner3',
    'door-lock-v2': 'corner',
}

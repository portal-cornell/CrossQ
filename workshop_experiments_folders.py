task_name_to_plot = {
    "right_arm_extend_wave_higher": "Right Arm Up",
    "left_arm_extend_wave_higher": "Left Arm Up",
    "right_arm_out": "Right Arm Out",
    "left_arm_out": "Left Arm Out",
    "both_arms_out": "Both Arms Out",
    "both_arms_down": "Both Arms Down"
}


joint_based_experiments_dict = {
    "right_arm_extend_wave_higher": {
        "ground_truth_baseline": {
            "Last-Joint": "/share/portal/hw575/CrossQ/train_logs/workshop_results/right_arm_extend_wave_higher/2024-10-06-111557_sb3_sac_envr=goal_only_euclidean_geom_xpos-t=right_arm_extend_wave_higher_rm=hand_engineered_nt=None"
        },
        "intermediate_10_frames": {
            "SDTW+": "/share/portal/hw575/CrossQ/train_logs/workshop_results/right_arm_extend_wave_higher/2024-10-06-010135_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_rm=soft_dtw_nt=intermediate_10_frames_exp-r+bonus",
            "SDTW": "/share/portal/hw575/CrossQ/train_logs/workshop_results/right_arm_extend_wave_higher/2024-10-07-214947_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_rm=soft_dtw_nt=intermediate_10_frames_exp-r",
            # "DTW+": "/share/portal/hw575/CrossQ/train_logs/workshop_results/right_arm_extend_wave_higher/2024-10-08-120751_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_rm=dtw_nt=intermediate_10_frames_exp-r+bonus",
            "DTW": "/share/portal/hw575/CrossQ/train_logs/workshop_results/right_arm_extend_wave_higher/2024-10-08-120748_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_rm=dtw_nt=intermediate_10_frames",
            "OT": "/share/portal/hw575/CrossQ/train_logs/workshop_results/right_arm_extend_wave_higher/2024-10-08-120726_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_rm=ot_nt=intermediate_10_frames",
        },
        # For workshop paper, we are no longer using 40 frames
        # 'intermediate_40_frames': {
        #     "SDTW+": "/share/portal/hw575/CrossQ/train_logs/workshop_results/right_arm_extend_wave_higher/2024-10-06-011603_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_rm=soft_dtw_nt=intermediate_40_frames_exp-r+bonus",
        #     "DTW": "/share/portal/hw575/CrossQ/train_logs/workshop_results/right_arm_extend_wave_higher/2024-10-06-011607_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_rm=dtw_nt=intermediate_40_frames",
        #     "OT": "/share/portal/hw575/CrossQ/train_logs/workshop_results/right_arm_extend_wave_higher/2024-10-06-011603_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_rm=ot_nt=intermediate_40_frames",
        # },
    },
    "left_arm_extend_wave_higher": {
        "ground_truth_baseline": {
            "Last-Joint": "/share/portal/hw575/CrossQ/train_logs/workshop_results/left_arm_extend_wave_higher/2024-10-07-215648_sb3_sac_envr=goal_only_euclidean_geom_xpos-t=left_arm_extend_wave_higher_rm=hand_engineered_nt=None"
        },
        "intermediate_10_frames": {
            "SDTW+": "/share/portal/hw575/CrossQ/train_logs/workshop_results/left_arm_extend_wave_higher/2024-10-08-012157_sb3_sac_envr=basic_r_geom_xpos-t=left_arm_extend_wave_higher_rm=soft_dtw_nt=intermediate_10_frames_exp-r+bonus",
            "SDTW": "/share/portal/hw575/CrossQ/train_logs/workshop_results/left_arm_extend_wave_higher/2024-10-07-220117_sb3_sac_envr=basic_r_geom_xpos-t=left_arm_extend_wave_higher_rm=soft_dtw_nt=intermediate_10_frames_exp-r", 
            # "DTW+": "/share/portal/hw575/CrossQ/train_logs/workshop_results/left_arm_extend_wave_higher/2024-10-08-121123_sb3_sac_envr=basic_r_geom_xpos-t=left_arm_extend_wave_higher_rm=dtw_nt=intermediate_10_frames_bonus",
            "DTW": "/share/portal/hw575/CrossQ/train_logs/workshop_results/left_arm_extend_wave_higher/2024-10-08-121001_sb3_sac_envr=basic_r_geom_xpos-t=left_arm_extend_wave_higher_rm=dtw_nt=intermediate_10_frames",
            "OT": "/share/portal/hw575/CrossQ/train_logs/workshop_results/left_arm_extend_wave_higher/2024-10-08-120956_sb3_sac_envr=basic_r_geom_xpos-t=left_arm_extend_wave_higher_rm=ot_nt=intermediate_10_frames"
        }
    },
    # Using seed 190 for training
    # "right_arm_out": {
    #     "ground_truth_baseline": {
    #         "Last-Joint": "/share/portal/hw575/CrossQ/train_logs/workshop_results/right_arm_out_seed=190/2024-10-09-011214_sb3_sac_envr=goal_only_euclidean_geom_xpos-t=right_arm_out_rm=hand_engineered_nt=None"
    #     },
    #     "intermediate_10_frames": {
    #         "SDTW+": "/share/portal/hw575/CrossQ/train_logs/workshop_results/right_arm_out_seed=190/2024-10-09-011247_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_out_rm=soft_dtw_nt=intermediate_10_frames_exp-r+bonus",
    #         "SDTW": "/share/portal/hw575/CrossQ/train_logs/workshop_results/right_arm_out_seed=190/2024-10-09-011332_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_out_rm=soft_dtw_nt=intermediate_10_frames_exp-r",
    #         "DTW+": "/share/portal/hw575/CrossQ/train_logs/workshop_results/right_arm_out_seed=190/2024-10-09-011406_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_out_rm=dtw_nt=intermediate_10_frames_bonus",
    #         "DTW": "/share/portal/hw575/CrossQ/train_logs/workshop_results/right_arm_out_seed=190/2024-10-09-011351_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_out_rm=dtw_nt=intermediate_10_frames",
    #         "OT": "/share/portal/hw575/CrossQ/train_logs/workshop_results/right_arm_out_seed=190/2024-10-09-011342_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_out_rm=ot_nt=intermediate_10_frames"
    #     }
    # },
    # Using seed 9 for training (same as other runs)
    "right_arm_out": {
        "ground_truth_baseline": {
            "Last-Joint": "/share/portal/hw575/CrossQ/train_logs/workshop_results/right_arm_out/2024-10-08-012519_sb3_sac_envr=goal_only_euclidean_geom_xpos-t=right_arm_out_rm=hand_engineered_nt=None"
        },
        "intermediate_10_frames": {
            "SDTW+": "/share/portal/hw575/CrossQ/train_logs/workshop_results/right_arm_out/2024-10-08-012222_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_out_rm=soft_dtw_nt=intermediate_10_frames_exp-r+bonus",
            "SDTW": "/share/portal/hw575/CrossQ/train_logs/workshop_results/right_arm_out/2024-10-07-220239_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_out_rm=soft_dtw_nt=intermediate_10_frames_exp-r",
            # "DTW+": "/share/portal/hw575/CrossQ/train_logs/workshop_results/right_arm_out/2024-10-08-121317_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_out_rm=dtw_nt=intermediate_10_frames_bonus",
            "DTW": "/share/portal/hw575/CrossQ/train_logs/workshop_results/right_arm_out/2024-10-08-121259_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_out_rm=dtw_nt=intermediate_10_frames",
            "OT": "/share/portal/hw575/CrossQ/train_logs/workshop_results/right_arm_out/2024-10-08-121254_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_out_rm=ot_nt=intermediate_10_frames"
        }
    },
    "left_arm_out": {
        "ground_truth_baseline": {
            "Last-Joint": "/share/portal/hw575/CrossQ/train_logs/workshop_results/left_arm_out/2024-10-06-111708_sb3_sac_envr=goal_only_euclidean_geom_xpos-t=left_arm_out_rm=hand_engineered_nt=None"
        },
        'intermediate_10_frames': {
            "SDTW+": "/share/portal/hw575/CrossQ/train_logs/workshop_results/left_arm_out/2024-10-06-010330_sb3_sac_envr=basic_r_geom_xpos-t=left_arm_out_rm=soft_dtw_nt=intermediate_10_frames_exp-r+bonus",
            "SDTW": "/share/portal/hw575/CrossQ/train_logs/workshop_results/left_arm_out/2024-10-07-215256_sb3_sac_envr=basic_r_geom_xpos-t=left_arm_out_rm=soft_dtw_nt=intermediate_10_frames_exp-r",
            # "DTW+": "/share/portal/hw575/CrossQ/train_logs/workshop_results/left_arm_out/2024-10-08-121532_sb3_sac_envr=basic_r_geom_xpos-t=left_arm_out_rm=dtw_nt=intermediate_10_frames_bonus",
            "DTW": "/share/portal/hw575/CrossQ/train_logs/workshop_results/left_arm_out/2024-10-08-121502_sb3_sac_envr=basic_r_geom_xpos-t=left_arm_out_rm=dtw_nt=intermediate_10_frames",
            "OT": "/share/portal/hw575/CrossQ/train_logs/workshop_results/left_arm_out/2024-10-08-121455_sb3_sac_envr=basic_r_geom_xpos-t=left_arm_out_rm=ot_nt=intermediate_10_frames"
        },
        # For workshop paper, we are no longer using 40 frames
        # 'intermediate_40_frames': {
        #     "SDTW+": "/share/portal/hw575/CrossQ/train_logs/workshop_results/2024-10-06-010748_sb3_sac_envr=basic_r_geom_xpos-t=left_arm_out_rm=soft_dtw_nt=intermediate_40_frames_exp-r+bonus",
        #     "DTW": "/share/portal/hw575/CrossQ/train_logs/workshop_results/2024-10-06-010805_sb3_sac_envr=basic_r_geom_xpos-t=left_arm_out_rm=dtw_nt=intermediate_40_frames",
        #     "OT": "/share/portal/hw575/CrossQ/train_logs/workshop_results/2024-10-06-010758_sb3_sac_envr=basic_r_geom_xpos-t=left_arm_out_rm=ot_nt=intermediate_40_frames"
        # }
    },
    "both_arms_out": {
        "ground_truth_baseline": {
            "Last-Joint": "/share/portal/hw575/CrossQ/train_logs/workshop_results/both_arms_out/2024-10-07-220337_sb3_sac_envr=goal_only_euclidean_geom_xpos-t=both_arms_out_rm=hand_engineered_nt=None"
        },
        "intermediate_10_frames": {
            "SDTW+": "/share/portal/hw575/CrossQ/train_logs/workshop_results/both_arms_out/2024-10-08-012315_sb3_sac_envr=basic_r_geom_xpos-t=both_arms_out_rm=soft_dtw_nt=intermediate_10_frames_exp-r+bonus",
            "SDTW": "/share/portal/hw575/CrossQ/train_logs/workshop_results/both_arms_out/2024-10-07-220400_sb3_sac_envr=basic_r_geom_xpos-t=both_arms_out_rm=soft_dtw_nt=intermediate_10_frames_exp-r",
            # "DTW+": "/share/portal/hw575/CrossQ/train_logs/workshop_results/both_arms_out/2024-10-08-121648_sb3_sac_envr=basic_r_geom_xpos-t=both_arms_out_rm=dtw_nt=intermediate_10_frames_bonus",
            "DTW": "/share/portal/hw575/CrossQ/train_logs/workshop_results/both_arms_out/2024-10-08-121637_sb3_sac_envr=basic_r_geom_xpos-t=both_arms_out_rm=dtw_nt=intermediate_10_frames",
            "OT": "/share/portal/hw575/CrossQ/train_logs/workshop_results/both_arms_out/2024-10-08-121627_sb3_sac_envr=basic_r_geom_xpos-t=both_arms_out_rm=ot_nt=intermediate_10_frames"
        }
    },
    "both_arms_down": {
        "ground_truth_baseline": {
            "Last-Joint": "/share/portal/hw575/CrossQ/train_logs/workshop_results/both_arms_down/2024-10-08-012937_sb3_sac_envr=goal_only_euclidean_geom_xpos-t=both_arms_down_rm=hand_engineered_nt=None"
        },
        "intermediate_10_frames": {
            "SDTW+": "/share/portal/hw575/CrossQ/train_logs/workshop_results/both_arms_down/2024-10-08-012327_sb3_sac_envr=basic_r_geom_xpos-t=both_arms_down_rm=soft_dtw_nt=intermediate_10_frames_exp-r+bonus",
            "SDTW": "/share/portal/hw575/CrossQ/train_logs/workshop_results/both_arms_down/2024-10-07-220647_sb3_sac_envr=basic_r_geom_xpos-t=both_arms_down_rm=soft_dtw_nt=intermediate_10_frames_exp-r",
            # "DTW+": "/share/portal/hw575/CrossQ/train_logs/workshop_results/both_arms_down/2024-10-08-122047_sb3_sac_envr=basic_r_geom_xpos-t=both_arms_down_rm=dtw_nt=intermediate_10_frames_bonus",
            "DTW": "/share/portal/hw575/CrossQ/train_logs/workshop_results/both_arms_down/2024-10-08-122046_sb3_sac_envr=basic_r_geom_xpos-t=both_arms_down_rm=dtw_nt=intermediate_10_frames",
            "OT": "/share/portal/hw575/CrossQ/train_logs/workshop_results/both_arms_down/2024-10-08-121812_sb3_sac_envr=basic_r_geom_xpos-t=both_arms_down_rm=ot_nt=intermediate_10_frames"
        }
    }
}


# Copy the format of joint_based_experiments_dict, but leave the path as empty string
visual_based_experiments_dict = {
    "right_arm_extend_wave_higher": {
        "ground_truth_baseline": {
            "Last-Joint": ""
        },
        "intermediate_10_frames": {
            "SDTW+": "",
            "SDTW": "",
            "DTW+": "",
            "DTW": "",
            "OT": "",
        },
    },
    "left_arm_extend_wave_higher": {
        "ground_truth_baseline": {
            "Last-Joint": ""
        },
        "intermediate_10_frames": {
            "SDTW+": "",
            "SDTW": "",
            "DTW+": "",
            "DTW": "",
            "OT": ""
        }
    },
    "right_arm_out": {
        "ground_truth_baseline": {
            "Last-Joint": ""
        },
        "intermediate_10_frames": {
            "SDTW+": "",
            "SDTW": "",
            "DTW+": "",
            "DTW": "",
            "OT": ""
        }
    },
    "left_arm_out": {
        "ground_truth_baseline": {
            "Last-Joint": ""
        },
        'intermediate_10_frames': {
            "SDTW+": "",
            "SDTW": "",
            "DTW+": "",
            "DTW": "",
            "OT": ""
        }
    },
    "both_arms_out": {
        "ground_truth_baseline": {
            "Last-Joint": ""
        },
        "intermediate_10_frames": {
            "SDTW+": "",
            "SDTW": "",
            "DTW+": "",
            "DTW": "",
            "OT": ""
        }
    },
    "both_arms_down": {
        "ground_truth_baseline": {
            "Last-Joint": ""
        },
        "intermediate_10_frames": {
            "SDTW+": "",
            "SDTW": "",
            "DTW+": "",
            "DTW": "",
            "OT": ""
        }
    }
}
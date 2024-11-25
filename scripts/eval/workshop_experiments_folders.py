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
            "DTW+": "/share/portal/hw575/CrossQ/train_logs/workshop_results/right_arm_extend_wave_higher/2024-10-08-120751_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_rm=dtw_nt=intermediate_10_frames_exp-r+bonus",
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
            "DTW+": "/share/portal/hw575/CrossQ/train_logs/workshop_results/left_arm_extend_wave_higher/2024-10-08-121123_sb3_sac_envr=basic_r_geom_xpos-t=left_arm_extend_wave_higher_rm=dtw_nt=intermediate_10_frames_bonus",
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
            "DTW+": "/share/portal/hw575/CrossQ/train_logs/workshop_results/right_arm_out/2024-10-08-121317_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_out_rm=dtw_nt=intermediate_10_frames_bonus",
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
            "DTW+": "/share/portal/hw575/CrossQ/train_logs/workshop_results/left_arm_out/2024-10-08-121532_sb3_sac_envr=basic_r_geom_xpos-t=left_arm_out_rm=dtw_nt=intermediate_10_frames_bonus",
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
            "DTW+": "/share/portal/hw575/CrossQ/train_logs/workshop_results/both_arms_out/2024-10-08-121648_sb3_sac_envr=basic_r_geom_xpos-t=both_arms_out_rm=dtw_nt=intermediate_10_frames_bonus",
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
            "DTW+": "/share/portal/hw575/CrossQ/train_logs/workshop_results/both_arms_down/2024-10-08-122047_sb3_sac_envr=basic_r_geom_xpos-t=both_arms_down_rm=dtw_nt=intermediate_10_frames_bonus",
            "DTW": "/share/portal/hw575/CrossQ/train_logs/workshop_results/both_arms_down/2024-10-08-122046_sb3_sac_envr=basic_r_geom_xpos-t=both_arms_down_rm=dtw_nt=intermediate_10_frames",
            "OT": "/share/portal/hw575/CrossQ/train_logs/workshop_results/both_arms_down/2024-10-08-121812_sb3_sac_envr=basic_r_geom_xpos-t=both_arms_down_rm=ot_nt=intermediate_10_frames"
        }
    }
}


# Visual rollout + visual reference + pre match confidence scaling
visual_based_experiments_dict = {
    "right_arm_extend_wave_higher": {
        "ground_truth_baseline": {
            "Last-Joint": "/share/portal/hw575/CrossQ/train_logs/workshop_results/right_arm_extend_wave_higher/2024-10-06-111557_sb3_sac_envr=goal_only_euclidean_geom_xpos-t=right_arm_extend_wave_higher_rm=hand_engineered_nt=None"
        },
        "intermediate_10_frames": {
            "SDTW+": "/share/portal/wph52/CrossQ/train_logs/2024-10-16-223632_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_mrm=soft_dtw_vrm=joint_pred_resnet_nt=None",
            "SDTW": "/share/portal/wph52/CrossQ/train_logs/2024-10-17-153840_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_mrm=soft_dtw_vrm=joint_pred_resnet_nt=None",
            "DTW+": "/share/portal/wph52/CrossQ/train_logs/2024-10-17-141316_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_mrm=dtw_vrm=joint_pred_resnet_nt=None",
            "DTW": "/share/portal/wph52/CrossQ/train_logs/2024-10-17-063357_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_mrm=dtw_vrm=joint_pred_resnet_nt=None",
            "OT": "/share/portal/wph52/CrossQ/train_logs/2024-10-17-023852_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_mrm=ot_vrm=joint_pred_resnet_nt=None",
            "RoboCLIP": "/share/portal/aw588/train_logs/roboclip/2024-10-12-200152_t=right_arm_extend_wave_higher_nt=sac",
            # "roboclip_ppo": "/share/portal/aw588/train_logs/roboclip/2024-10-13-134730_t=right_arm_extend_wave_higher_nt=ppo",
        },
    },
    "left_arm_extend_wave_higher": {
        "ground_truth_baseline": {
            "Last-Joint": "/share/portal/hw575/CrossQ/train_logs/workshop_results/left_arm_extend_wave_higher/2024-10-07-215648_sb3_sac_envr=goal_only_euclidean_geom_xpos-t=left_arm_extend_wave_higher_rm=hand_engineered_nt=None"
        },
        "intermediate_10_frames": {
            "SDTW+": "/share/portal/wph52/CrossQ/train_logs/2024-10-16-223435_sb3_sac_envr=basic_r_geom_xpos-t=left_arm_extend_wave_higher_mrm=soft_dtw_vrm=joint_pred_resnet_nt=None",
            "SDTW": "/share/portal/wph52/CrossQ/train_logs/2024-10-17-101114_sb3_sac_envr=basic_r_geom_xpos-t=left_arm_extend_wave_higher_mrm=soft_dtw_vrm=joint_pred_resnet_nt=None",
            "DTW+": "/share/portal/wph52/CrossQ/train_logs/2024-10-17-140010_sb3_sac_envr=basic_r_geom_xpos-t=left_arm_extend_wave_higher_mrm=dtw_vrm=joint_pred_resnet_nt=None",
            "DTW": "/share/portal/wph52/CrossQ/train_logs/2024-10-17-062307_sb3_sac_envr=basic_r_geom_xpos-t=left_arm_extend_wave_higher_mrm=dtw_vrm=joint_pred_resnet_nt=None",
            "OT": "/share/portal/wph52/CrossQ/train_logs/2024-10-17-022553_sb3_sac_envr=basic_r_geom_xpos-t=left_arm_extend_wave_higher_mrm=ot_vrm=joint_pred_resnet_nt=None",
            "RoboCLIP": "/share/portal/aw588/train_logs/roboclip/2024-10-12-154406_t=left_arm_extend_wave_higher_nt=sac",
            # "roboclip_ppo": "/share/portal/aw588/train_logs/roboclip/2024-10-12-205955_t=left_arm_extend_wave_higher_nt=ppo",
        }
    },
    "right_arm_out": {
        "ground_truth_baseline": {
            "Last-Joint": "/share/portal/hw575/CrossQ/train_logs/workshop_results/right_arm_out/2024-10-08-012519_sb3_sac_envr=goal_only_euclidean_geom_xpos-t=right_arm_out_rm=hand_engineered_nt=None"
        },
        "intermediate_10_frames": {
            "SDTW+": "/share/portal/wph52/CrossQ/train_logs/2024-10-16-223524_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_out_mrm=soft_dtw_vrm=joint_pred_resnet_nt=None",
            "SDTW": "/share/portal/wph52/CrossQ/train_logs/2024-10-17-153727_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_out_mrm=soft_dtw_vrm=joint_pred_resnet_nt=None",
            "DTW+": "/share/portal/wph52/CrossQ/train_logs/2024-10-17-083610_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_out_mrm=dtw_vrm=joint_pred_resnet_nt=None",
            "DTW": "/share/portal/wph52/CrossQ/train_logs/2024-10-17-033521_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_out_mrm=dtw_vrm=joint_pred_resnet_nt=None",
            "OT": "/share/portal/wph52/CrossQ/train_logs/2024-10-17-010537_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_out_mrm=ot_vrm=joint_pred_resnet_nt=None",
            "RoboCLIP": "/share/portal/aw588/train_logs/roboclip/2024-10-12-154616_t=right_arm_out_nt=sac",
            # "roboclip_ppo": "/share/portal/aw588/train_logs/roboclip/2024-10-13-134739_t=right_arm_out_nt=ppo"
        }
    },
    "left_arm_out": {
        "ground_truth_baseline": {
            "Last-Joint": "/share/portal/hw575/CrossQ/train_logs/workshop_results/left_arm_out/2024-10-06-111708_sb3_sac_envr=goal_only_euclidean_geom_xpos-t=left_arm_out_rm=hand_engineered_nt=None"
        },
        'intermediate_10_frames': {
            "SDTW+": "/share/portal/wph52/CrossQ/train_logs/2024-10-16-223509_sb3_sac_envr=basic_r_geom_xpos-t=left_arm_out_mrm=soft_dtw_vrm=joint_pred_resnet_nt=None",
            "SDTW": "/share/portal/wph52/CrossQ/train_logs/2024-10-17-153754_sb3_sac_envr=basic_r_geom_xpos-t=left_arm_out_mrm=soft_dtw_vrm=joint_pred_resnet_nt=None",
            "DTW+": "/share/portal/wph52/CrossQ/train_logs/2024-10-17-141233_sb3_sac_envr=basic_r_geom_xpos-t=left_arm_out_mrm=dtw_vrm=joint_pred_resnet_nt=None",
            "DTW": "/share/portal/wph52/CrossQ/train_logs/2024-10-17-063208_sb3_sac_envr=basic_r_geom_xpos-t=left_arm_out_mrm=dtw_vrm=joint_pred_resnet_nt=None",
            "OT": "/share/portal/wph52/CrossQ/train_logs/2024-10-17-023701_sb3_sac_envr=basic_r_geom_xpos-t=left_arm_out_mrm=ot_vrm=joint_pred_resnet_nt=None",
            "RoboCLIP": "/share/portal/aw588/train_logs/roboclip/2024-10-12-205825_t=left_arm_out_nt=sac",
            # "roboclip_ppo": "/share/portal/aw588/train_logs/roboclip/2024-10-13-134746_t=left_arm_out_nt=ppo",
        }
    },
    "both_arms_out": {
        "ground_truth_baseline": {
            "Last-Joint": "/share/portal/hw575/CrossQ/train_logs/workshop_results/both_arms_out/2024-10-07-220337_sb3_sac_envr=goal_only_euclidean_geom_xpos-t=both_arms_out_rm=hand_engineered_nt=None"
        },
        "intermediate_10_frames": 
        {
            "SDTW+": "/share/portal/wph52/CrossQ/train_logs/2024-10-16-223418_sb3_sac_envr=basic_r_geom_xpos-t=both_arms_out_mrm=soft_dtw_vrm=joint_pred_resnet_nt=None",
            "SDTW": "/share/portal/wph52/CrossQ/train_logs/2024-10-17-153857_sb3_sac_envr=basic_r_geom_xpos-t=both_arms_out_mrm=soft_dtw_vrm=joint_pred_resnet_nt=None",
            "DTW+": "/share/portal/wph52/CrossQ/train_logs/2024-10-17-154837_sb3_sac_envr=basic_r_geom_xpos-t=both_arms_out_mrm=dtw_vrm=joint_pred_resnet_nt=None",
            "DTW": "/share/portal/wph52/CrossQ/train_logs/2024-10-17-075453_sb3_sac_envr=basic_r_geom_xpos-t=both_arms_out_mrm=dtw_vrm=joint_pred_resnet_nt=None",
            "OT": "/share/portal/wph52/CrossQ/train_logs/2024-10-17-030639_sb3_sac_envr=basic_r_geom_xpos-t=both_arms_out_mrm=ot_vrm=joint_pred_resnet_nt=None",
            "RoboCLIP": "/share/portal/aw588/train_logs/roboclip/2024-10-12-154244_t=both_arms_out_nt=sac",
            # "roboclip_ppo": "/share/portal/aw588/train_logs/roboclip/2024-10-12-205908_t=both_arms_out_nt=ppo",
        }
    },
    "both_arms_down": {
        "ground_truth_baseline": {
            "Last-Joint": "/share/portal/hw575/CrossQ/train_logs/workshop_results/both_arms_down/2024-10-08-012937_sb3_sac_envr=goal_only_euclidean_geom_xpos-t=both_arms_down_rm=hand_engineered_nt=None"            
        },
        "intermediate_10_frames": 
        {
            "SDTW+": "/share/portal/wph52/CrossQ/train_logs/2024-10-16-223404_sb3_sac_envr=basic_r_geom_xpos-t=both_arms_down_mrm=soft_dtw_vrm=joint_pred_resnet_nt=None",
            "SDTW": "/share/portal/wph52/CrossQ/train_logs/2024-10-17-153943_sb3_sac_envr=basic_r_geom_xpos-t=both_arms_down_mrm=soft_dtw_vrm=joint_pred_resnet_nt=None",
            "DTW+": "/share/portal/wph52/CrossQ/train_logs/2024-10-17-154834_sb3_sac_envr=basic_r_geom_xpos-t=both_arms_down_mrm=dtw_vrm=joint_pred_resnet_nt=None",
            "DTW": "/share/portal/wph52/CrossQ/train_logs/2024-10-17-075508_sb3_sac_envr=basic_r_geom_xpos-t=both_arms_down_mrm=dtw_vrm=joint_pred_resnet_nt=None",
            "OT": "/share/portal/wph52/CrossQ/train_logs/2024-10-17-030921_sb3_sac_envr=basic_r_geom_xpos-t=both_arms_down_mrm=ot_vrm=joint_pred_resnet_nt=None",
            "RoboCLIP": "/share/portal/aw588/train_logs/roboclip/2024-10-12-154447_t=both_arms_down_nt=sac",
            # "roboclip_ppo": "/share/portal/aw588/train_logs/roboclip/2024-10-13-134733_t=both_arms_down_nt=ppo",
        }
    }
}

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

# Copy the format of joint_based_experiments_dict, but leave the path as empty string
visual_based_experiments_dict = {
    "right_arm_extend_wave_higher": {
        "ground_truth_baseline": {
            "Last-Joint": "/share/portal/hw575/CrossQ/train_logs/workshop_results/right_arm_extend_wave_higher/2024-10-06-111557_sb3_sac_envr=goal_only_euclidean_geom_xpos-t=right_arm_extend_wave_higher_rm=hand_engineered_nt=None"
        },
        "intermediate_10_frames": {
            "SDTW+": "/share/portal/wph52/CrossQ/train_logs/2024-10-08-125634_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_mrm=soft_dtw_vrm=joint_pred_resnet_nt=None",
            "SDTW": "/share/portal/wph52/CrossQ/train_logs/2024-10-08-171003_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_mrm=soft_dtw_vrm=joint_pred_resnet_nt=None",
            "DTW+": "/share/portal/wph52/CrossQ/train_logs/2024-10-09-014029_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_mrm=dtw_vrm=joint_pred_resnet_nt=None",
            "DTW": "/share/portal/wph52/CrossQ/train_logs/2024-10-08-212449_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_mrm=dtw_vrm=joint_pred_resnet_nt=None",
            "OT": "",
        },
    },
    "left_arm_extend_wave_higher": {
        "ground_truth_baseline": {
            "Last-Joint": "/share/portal/hw575/CrossQ/train_logs/workshop_results/left_arm_extend_wave_higher/2024-10-07-215648_sb3_sac_envr=goal_only_euclidean_geom_xpos-t=left_arm_extend_wave_higher_rm=hand_engineered_nt=None"
        },
        "intermediate_10_frames": {
            "SDTW+": "/share/portal/wph52/CrossQ/train_logs/2024-10-08-125553_sb3_sac_envr=basic_r_geom_xpos-t=left_arm_extend_wave_higher_mrm=soft_dtw_vrm=joint_pred_resnet_nt=None",
            "SDTW": "/share/portal/wph52/CrossQ/train_logs/2024-10-08-170514_sb3_sac_envr=basic_r_geom_xpos-t=left_arm_extend_wave_higher_mrm=soft_dtw_vrm=joint_pred_resnet_nt=None",
            "DTW+": "/share/portal/wph52/CrossQ/train_logs/2024-10-09-011740_sb3_sac_envr=basic_r_geom_xpos-t=left_arm_extend_wave_higher_mrm=dtw_vrm=joint_pred_resnet_nt=None",
            "DTW": "/share/portal/wph52/CrossQ/train_logs/2024-10-08-211133_sb3_sac_envr=basic_r_geom_xpos-t=left_arm_extend_wave_higher_mrm=dtw_vrm=joint_pred_resnet_nt=None",
            "OT": ""
        }
    },
    "right_arm_out": {
        "ground_truth_baseline": {
            "Last-Joint": "/share/portal/hw575/CrossQ/train_logs/workshop_results/right_arm_out/2024-10-08-012519_sb3_sac_envr=goal_only_euclidean_geom_xpos-t=right_arm_out_rm=hand_engineered_nt=None"
        },
        "intermediate_10_frames": {
            "SDTW+": "/share/portal/wph52/CrossQ/train_logs/2024-10-08-125656_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_out_mrm=soft_dtw_vrm=joint_pred_resnet_nt=None",
            "SDTW": "/share/portal/wph52/CrossQ/train_logs/2024-10-08-171840_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_out_mrm=soft_dtw_vrm=joint_pred_resnet_nt=None",
            "DTW+": "/share/portal/wph52/CrossQ/train_logs/2024-10-09-015007_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_out_mrm=dtw_vrm=joint_pred_resnet_nt=None",
            "DTW": "/share/portal/wph52/CrossQ/train_logs/2024-10-08-213341_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_out_mrm=dtw_vrm=joint_pred_resnet_nt=None",
            "OT": ""
        }
    },
    "left_arm_out": {
        "ground_truth_baseline": {
            "Last-Joint": "/share/portal/hw575/CrossQ/train_logs/workshop_results/left_arm_out/2024-10-06-111708_sb3_sac_envr=goal_only_euclidean_geom_xpos-t=left_arm_out_rm=hand_engineered_nt=None"
        },
        'intermediate_10_frames': {
            "SDTW+": "/share/portal/wph52/CrossQ/train_logs/2024-10-08-125617_sb3_sac_envr=basic_r_geom_xpos-t=left_arm_out_mrm=soft_dtw_vrm=joint_pred_resnet_nt=None",
            "SDTW": "/share/portal/wph52/CrossQ/train_logs/2024-10-08-171654_sb3_sac_envr=basic_r_geom_xpos-t=left_arm_out_mrm=soft_dtw_vrm=joint_pred_resnet_nt=None",
            "DTW+": "/share/portal/wph52/CrossQ/train_logs/2024-10-09-014738_sb3_sac_envr=basic_r_geom_xpos-t=left_arm_out_mrm=dtw_vrm=joint_pred_resnet_nt=None",
            "DTW": "/share/portal/wph52/CrossQ/train_logs/2024-10-08-213139_sb3_sac_envr=basic_r_geom_xpos-t=left_arm_out_mrm=dtw_vrm=joint_pred_resnet_nt=None",
            "OT": ""
        }
    },
    "both_arms_out": {
        "ground_truth_baseline": {
            "Last-Joint": "/share/portal/hw575/CrossQ/train_logs/workshop_results/both_arms_out/2024-10-07-220337_sb3_sac_envr=goal_only_euclidean_geom_xpos-t=both_arms_out_rm=hand_engineered_nt=None"
        },
        "intermediate_10_frames": {
            "SDTW+": "/share/portal/wph52/CrossQ/train_logs/2024-10-08-125543_sb3_sac_envr=basic_r_geom_xpos-t=both_arms_out_mrm=soft_dtw_vrm=joint_pred_resnet_nt=None",
            "SDTW": "/share/portal/wph52/CrossQ/train_logs/2024-10-08-170946_sb3_sac_envr=basic_r_geom_xpos-t=both_arms_out_mrm=soft_dtw_vrm=joint_pred_resnet_nt=None",
            "DTW+": "/share/portal/wph52/CrossQ/train_logs/2024-10-09-014029_sb3_sac_envr=basic_r_geom_xpos-t=both_arms_out_mrm=dtw_vrm=joint_pred_resnet_nt=None",
            "DTW": "/share/portal/wph52/CrossQ/train_logs/2024-10-08-212446_sb3_sac_envr=basic_r_geom_xpos-t=both_arms_out_mrm=dtw_vrm=joint_pred_resnet_nt=None",
            "OT": ""
        }
    },
    "both_arms_down": {
        "ground_truth_baseline": {
            "Last-Joint": "/share/portal/hw575/CrossQ/train_logs/workshop_results/both_arms_down/2024-10-08-012937_sb3_sac_envr=goal_only_euclidean_geom_xpos-t=both_arms_down_rm=hand_engineered_nt=None"            
        },
        "intermediate_10_frames": {
            "SDTW+": "/share/portal/wph52/CrossQ/train_logs/2024-10-08-125918_sb3_sac_envr=basic_r_geom_xpos-t=both_arms_down_mrm=soft_dtw_vrm=joint_pred_resnet_nt=None",
            "SDTW": "/share/portal/wph52/CrossQ/train_logs/2024-10-08-170251_sb3_sac_envr=basic_r_geom_xpos-t=both_arms_down_mrm=soft_dtw_vrm=joint_pred_resnet_nt=None",
            "DTW+": "/share/portal/wph52/CrossQ/train_logs/2024-10-09-005311_sb3_sac_envr=basic_r_geom_xpos-t=both_arms_down_mrm=dtw_vrm=joint_pred_resnet_nt=None",
            "DTW": "/share/portal/wph52/CrossQ/train_logs/2024-10-08-205723_sb3_sac_envr=basic_r_geom_xpos-t=both_arms_down_mrm=dtw_vrm=joint_pred_resnet_nt=None",
            "OT": ""
        }
    }
}


# Copy the format of joint_based_experiments_dict, but leave the path as empty string
visual_rollout_gt_reference_experiments_dict = {
    "right_arm_extend_wave_higher": {
        "ground_truth_baseline": {
            "Last-Joint": "/share/portal/hw575/CrossQ/train_logs/workshop_results/right_arm_extend_wave_higher/2024-10-06-111557_sb3_sac_envr=goal_only_euclidean_geom_xpos-t=right_arm_extend_wave_higher_rm=hand_engineered_nt=None"
        },
        "intermediate_10_frames": {
            "SDTW+": "/share/portal/wph52/CrossQ/train_logs/2024-10-09-215538_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_mrm=soft_dtw_vrm=joint_pred_resnet_nt=None",
            "SDTW": "/share/portal/wph52/CrossQ/train_logs/2024-10-10-045016_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_mrm=soft_dtw_vrm=joint_pred_resnet_nt=None",
            "DTW+": "/share/portal/wph52/CrossQ/train_logs/2024-10-10-183332_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_mrm=dtw_vrm=joint_pred_resnet_nt=None",
            "DTW": "/share/portal/wph52/CrossQ/train_logs/2024-10-10-090904_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_mrm=dtw_vrm=joint_pred_resnet_nt=None",
            "OT": "/share/portal/wph52/CrossQ/train_logs/2024-10-10-003555_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_mrm=ot_vrm=joint_pred_resnet_nt=None",
        },
    },
    "left_arm_extend_wave_higher": {
        "ground_truth_baseline": {
            "Last-Joint": "/share/portal/hw575/CrossQ/train_logs/workshop_results/left_arm_extend_wave_higher/2024-10-07-215648_sb3_sac_envr=goal_only_euclidean_geom_xpos-t=left_arm_extend_wave_higher_rm=hand_engineered_nt=None"
        },
        "intermediate_10_frames": {
            "SDTW+": "/share/portal/wph52/CrossQ/train_logs/2024-10-09-215448_sb3_sac_envr=basic_r_geom_xpos-t=left_arm_extend_wave_higher_mrm=soft_dtw_vrm=joint_pred_resnet_nt=None",
            "SDTW": "/share/portal/wph52/CrossQ/train_logs/2024-10-10-044801_sb3_sac_envr=basic_r_geom_xpos-t=left_arm_extend_wave_higher_mrm=soft_dtw_vrm=joint_pred_resnet_nt=None",
            "DTW+": "/share/portal/wph52/CrossQ/train_logs/2024-10-10-183815_sb3_sac_envr=basic_r_geom_xpos-t=left_arm_extend_wave_higher_mrm=dtw_vrm=joint_pred_resnet_nt=None",
            "DTW": "/share/portal/wph52/CrossQ/train_logs/2024-10-10-090651_sb3_sac_envr=basic_r_geom_xpos-t=left_arm_extend_wave_higher_mrm=dtw_vrm=joint_pred_resnet_nt=None",
            "OT": "/share/portal/wph52/CrossQ/train_logs/2024-10-10-003425_sb3_sac_envr=basic_r_geom_xpos-t=left_arm_extend_wave_higher_mrm=ot_vrm=joint_pred_resnet_nt=None"
        }
    },
    "right_arm_out": {
        "ground_truth_baseline": {
            "Last-Joint": "/share/portal/hw575/CrossQ/train_logs/workshop_results/right_arm_out/2024-10-08-012519_sb3_sac_envr=goal_only_euclidean_geom_xpos-t=right_arm_out_rm=hand_engineered_nt=None"
        },
        "intermediate_10_frames": {
            "SDTW+": "/share/portal/wph52/CrossQ/train_logs/2024-10-09-215440_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_out_mrm=soft_dtw_vrm=joint_pred_resnet_nt=None",
            "SDTW": "/share/portal/wph52/CrossQ/train_logs/2024-10-10-050147_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_out_mrm=soft_dtw_vrm=joint_pred_resnet_nt=None",
            "DTW+": "/share/portal/wph52/CrossQ/train_logs/2024-10-10-183653_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_out_mrm=dtw_vrm=joint_pred_resnet_nt=None",
            "DTW": "/share/portal/wph52/CrossQ/train_logs/2024-10-10-094722_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_out_mrm=dtw_vrm=joint_pred_resnet_nt=None",
            "OT": "/share/portal/wph52/CrossQ/train_logs/2024-10-10-002357_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_out_mrm=ot_vrm=joint_pred_resnet_nt=None"
        }
    },
    "left_arm_out": {
        "ground_truth_baseline": {
            "Last-Joint": "/share/portal/hw575/CrossQ/train_logs/workshop_results/left_arm_out/2024-10-06-111708_sb3_sac_envr=goal_only_euclidean_geom_xpos-t=left_arm_out_rm=hand_engineered_nt=None"
        },
        'intermediate_10_frames': {
            "SDTW+": "/share/portal/wph52/CrossQ/train_logs/2024-10-09-160344_sb3_sac_envr=basic_r_geom_xpos-t=left_arm_out_mrm=soft_dtw_vrm=joint_pred_resnet_nt=None",
            "SDTW": "/share/portal/wph52/CrossQ/train_logs/2024-10-10-044621_sb3_sac_envr=basic_r_geom_xpos-t=left_arm_out_mrm=soft_dtw_vrm=joint_pred_resnet_nt=None",
            "DTW+": "/share/portal/wph52/CrossQ/train_logs/2024-10-10-183706_sb3_sac_envr=basic_r_geom_xpos-t=left_arm_out_mrm=dtw_vrm=joint_pred_resnet_nt=None",
            "DTW": "/share/portal/wph52/CrossQ/train_logs/2024-10-10-090020_sb3_sac_envr=basic_r_geom_xpos-t=left_arm_out_mrm=dtw_vrm=joint_pred_resnet_nt=None",
            "OT": "/share/portal/wph52/CrossQ/train_logs/2024-10-10-003242_sb3_sac_envr=basic_r_geom_xpos-t=left_arm_out_mrm=ot_vrm=joint_pred_resnet_nt=None"
        }
    },
    "both_arms_out": {
        "ground_truth_baseline": {
            "Last-Joint": "/share/portal/hw575/CrossQ/train_logs/workshop_results/both_arms_out/2024-10-07-220337_sb3_sac_envr=goal_only_euclidean_geom_xpos-t=both_arms_out_rm=hand_engineered_nt=None"
        },
        "intermediate_10_frames": {
            "SDTW+": "/share/portal/wph52/CrossQ/train_logs/2024-10-09-160341_sb3_sac_envr=basic_r_geom_xpos-t=both_arms_out_mrm=soft_dtw_vrm=joint_pred_resnet_nt=None",
            "SDTW": "/share/portal/wph52/CrossQ/train_logs/2024-10-10-044628_sb3_sac_envr=basic_r_geom_xpos-t=both_arms_out_mrm=soft_dtw_vrm=joint_pred_resnet_nt=None",
            "DTW+": "/share/portal/wph52/CrossQ/train_logs/2024-10-10-183852_sb3_sac_envr=basic_r_geom_xpos-t=both_arms_out_mrm=dtw_vrm=joint_pred_resnet_nt=None",
            "DTW": "/share/portal/wph52/CrossQ/train_logs/2024-10-10-090020_sb3_sac_envr=basic_r_geom_xpos-t=both_arms_out_mrm=dtw_vrm=joint_pred_resnet_nt=None",
            "OT": "/share/portal/wph52/CrossQ/train_logs/2024-10-10-003330_sb3_sac_envr=basic_r_geom_xpos-t=both_arms_out_mrm=ot_vrm=joint_pred_resnet_nt=None"
        }
    },
    "both_arms_down": {
        "ground_truth_baseline": {
            "Last-Joint": "/share/portal/hw575/CrossQ/train_logs/workshop_results/both_arms_down/2024-10-08-012937_sb3_sac_envr=goal_only_euclidean_geom_xpos-t=both_arms_down_rm=hand_engineered_nt=None"            
        },
        "intermediate_10_frames": {
            "SDTW+": "/share/portal/wph52/CrossQ/train_logs/2024-10-09-215550_sb3_sac_envr=basic_r_geom_xpos-t=both_arms_down_mrm=soft_dtw_vrm=joint_pred_resnet_nt=None",
            "SDTW": "/share/portal/wph52/CrossQ/train_logs/2024-10-10-045714_sb3_sac_envr=basic_r_geom_xpos-t=both_arms_down_mrm=soft_dtw_vrm=joint_pred_resnet_nt=None",
            "DTW+": "/share/portal/wph52/CrossQ/train_logs/2024-10-10-184104_sb3_sac_envr=basic_r_geom_xpos-t=both_arms_down_mrm=dtw_vrm=joint_pred_resnet_nt=None",
            "DTW": "/share/portal/wph52/CrossQ/train_logs/2024-10-10-094249_sb3_sac_envr=basic_r_geom_xpos-t=both_arms_down_mrm=dtw_vrm=joint_pred_resnet_nt=None",
            "OT": "/share/portal/wph52/CrossQ/train_logs/2024-10-10-215207_sb3_sac_envr=basic_r_geom_xpos-t=both_arms_down_mrm=ot_vrm=joint_pred_resnet_nt=None"
        }
    }
}


experiments_dict = {
    'right_arm_extend_wave_higher': {
        'ground_truth_baseline': {
            "ground_truth": "/share/portal/hw575/CrossQ/train_logs/workshop_results/2024-10-06-111557_sb3_sac_envr=goal_only_euclidean_geom_xpos-t=right_arm_extend_wave_higher_rm=hand_engineered_nt=None"
        },
        'intermediate_10_frames': {
            "sdtw+": "/share/portal/hw575/CrossQ/train_logs/workshop_results/2024-10-06-010135_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_rm=soft_dtw_nt=intermediate_10_frames_exp-r+bonus",
            "dtw": "/share/portal/hw575/CrossQ/train_logs/workshop_results/2024-10-06-010149_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_rm=dtw_nt=intermediate_10_frames",
            "ot": "/share/portal/hw575/CrossQ/train_logs/workshop_results/2024-10-06-010149_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_rm=ot_nt=intermediate_10_frames",
        },
        'intermediate_40_frames': {
            "sdtw+": "/share/portal/hw575/CrossQ/train_logs/workshop_results/2024-10-06-011603_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_rm=soft_dtw_nt=intermediate_40_frames_exp-r+bonus",
            "dtw": "/share/portal/hw575/CrossQ/train_logs/workshop_results/2024-10-06-011607_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_rm=dtw_nt=intermediate_40_frames",
            "ot": "/share/portal/hw575/CrossQ/train_logs/workshop_results/2024-10-06-011603_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_rm=ot_nt=intermediate_40_frames",
        },
    },
    "left_arm_out": {
        'ground_truth_baseline': {
            "ground_truth": "train_logs/workshop_results/2024-10-06-111708_sb3_sac_envr=goal_only_euclidean_geom_xpos-t=left_arm_out_rm=hand_engineered_nt=None"
        },
        'intermediate_10_frames': {
            "sdtw+": "/share/portal/hw575/CrossQ/train_logs/workshop_results/2024-10-06-010330_sb3_sac_envr=basic_r_geom_xpos-t=left_arm_out_rm=soft_dtw_nt=intermediate_10_frames_exp-r+bonus",
            "dtw": "/share/portal/hw575/CrossQ/train_logs/workshop_results/2024-10-06-010744_sb3_sac_envr=basic_r_geom_xpos-t=left_arm_out_rm=dtw_nt=intermediate_10_frames",
            "ot": "/share/portal/hw575/CrossQ/train_logs/workshop_results/2024-10-06-010334_sb3_sac_envr=basic_r_geom_xpos-t=left_arm_out_rm=ot_nt=intermediate_10_frames"
        },
        'intermediate_40_frames': {
            "sdtw+": "/share/portal/hw575/CrossQ/train_logs/workshop_results/2024-10-06-010748_sb3_sac_envr=basic_r_geom_xpos-t=left_arm_out_rm=soft_dtw_nt=intermediate_40_frames_exp-r+bonus",
            "dtw": "/share/portal/hw575/CrossQ/train_logs/workshop_results/2024-10-06-010805_sb3_sac_envr=basic_r_geom_xpos-t=left_arm_out_rm=dtw_nt=intermediate_40_frames",
            "ot": "/share/portal/hw575/CrossQ/train_logs/workshop_results/2024-10-06-010758_sb3_sac_envr=basic_r_geom_xpos-t=left_arm_out_rm=ot_nt=intermediate_40_frames"
        }
    }
}
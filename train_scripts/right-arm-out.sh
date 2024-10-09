python -m train "visual_reward_model=joint_pred_resnet" "matching_reward_model=optimal_transport" "compute.n_gpu_workers=1" "logging.wandb_tags=['resnet', 'intermediate_10_frames', 'geom_xpos', 'right_arm_out', 'ot', 'reco', '2M', 'visual_ref']" "env.task_name=right_arm_out" "env.reward_type=basic_r_geom_xpos" "visual_reward_model.use_image_for_ref=True" "matching_reward_model.seq_name=intermediate_10_frames" "logging.wandb_mode=online" 

# python -m train "visual_reward_model=joint_pred_resnet" "matching_reward_model=soft_dtw" "compute.n_gpu_workers=1" "logging.wandb_tags=['resnet', 'intermediate_10_frames', 'geom_xpos', 'right_arm_out', 'sdtw+', 'reco', '2M', 'visual_ref']" "env.task_name=right_arm_out" "env.reward_type=basic_r_geom_xpos" "visual_reward_model.use_image_for_ref=True" "matching_reward_model.seq_name=intermediate_10_frames" "matching_reward_model.post_processing_method=['exp_reward', 'stage_reward_based_on_last_state']" "logging.wandb_mode=online" 

# python -m train "visual_reward_model=joint_pred_resnet" "matching_reward_model=soft_dtw" "compute.n_gpu_workers=1" "logging.wandb_tags=['resnet', 'intermediate_10_frames', 'geom_xpos', 'right_arm_out', 'sdtw', 'reco', '2M', 'visual_ref']" "env.task_name=right_arm_out" "env.reward_type=basic_r_geom_xpos" "visual_reward_model.use_image_for_ref=True" "matching_reward_model.seq_name=intermediate_10_frames" "matching_reward_model.post_processing_method=['exp_reward']" "logging.wandb_mode=online" 

# python -m train "visual_reward_model=joint_pred_resnet" "matching_reward_model=dtw" "compute.n_gpu_workers=1" "logging.wandb_tags=['resnet', 'intermediate_10_frames', 'geom_xpos', 'right_arm_out', 'dtw', 'reco', '2M', 'visual_ref']" "env.task_name=right_arm_out" "env.reward_type=basic_r_geom_xpos" "visual_reward_model.use_image_for_ref=True" "matching_reward_model.seq_name=intermediate_10_frames" "logging.wandb_mode=online" 

# python -m train "visual_reward_model=joint_pred_resnet" "matching_reward_model=dtw" "compute.n_gpu_workers=1" "logging.wandb_tags=['resnet', 'intermediate_10_frames', 'geom_xpos', 'right_arm_out', 'dtw+', 'reco', '2M', 'visual_ref']" "env.task_name=right_arm_out" "env.reward_type=basic_r_geom_xpos" "visual_reward_model.use_image_for_ref=True" "matching_reward_model.seq_name=intermediate_10_frames" "logging.wandb_mode=online" "+matching_reward_model.post_processing_method=['stage_reward_based_on_last_state']"

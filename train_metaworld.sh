############### BUTTON PRESS ###############

# python train.py env=Metaworld env.env_reward_type='none' env.task_name='button-press-v2-goal-hidden' reward_model='dtw' reward_model.seq_name='rl_expert_20_frames' reward_model.cost_fn='euclidean' 'logging.wandb_tags=["gt_ref"]'

# python train.py env=Metaworld env.env_reward_type='none' env.task_name='button-press-v2-goal-hidden' env.temporal_encoding=true reward_model='soft_dtw_plus' reward_model.seq_name='rl_expert_20_frames' reward_model.cost_fn='euclidean' 'logging.wandb_tags=["gt_ref"]'

# python train.py env=Metaworld env.env_reward_type='none' env.task_name='button-press-v2-goal-hidden' env.temporal_encoding=true reward_model='even_distribution' reward_model.seq_name='rl_expert_20_frames' reward_model.cost_fn='euclidean' 'logging.wandb_tags=["gt_ref"]'

# python train.py env=Metaworld env.env_reward_type='none' env.task_name='button-press-v2-goal-hidden' env.temporal_encoding=true reward_model='prob_ranked' reward_model.seq_name='rl_expert_20_frames' reward_model.cost_fn='euclidean' 'logging.wandb_tags=["gt_ref", "scale_100"]'

# python train.py env=Metaworld env.env_reward_type='sparse' env.task_name='button-press-v2-goal-observable' env.temporal_encoding=true


############### DOOR CLOSE ###############

# python train.py env=Metaworld env.env_reward_type='none' env.task_name='door-close-v2-goal-hidden' env.temporal_encoding=true reward_model='even_distribution' reward_model.seq_name='rl_expert_20_frames' reward_model.cost_fn='euclidean' 'logging.wandb_tags=["gt_ref"]'

python train.py env=Metaworld env.env_reward_type='none' env.task_name='door-close-v2-goal-hidden' env.temporal_encoding=true reward_model='prob_reward' reward_model.seq_name='rl_expert_20_frames' reward_model.cost_fn='euclidean' 'logging.wandb_tags=["gt_ref", "scale_100"]'
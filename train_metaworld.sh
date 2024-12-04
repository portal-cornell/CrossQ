# python train.py env=Metaworld env.env_reward_type='none' env.task_name='button-press-v2-goal-hidden' reward_model='dtw' reward_model.seq_name='rl_expert_20_frames' reward_model.cost_fn='euclidean' 'logging.wandb_tags=["gt_ref"]'

python train.py env=Metaworld env.env_reward_type='none' env.task_name='button-press-v2-goal-hidden' reward_model='soft_dtw_plus' reward_model.seq_name='rl_expert_20_frames' reward_model.cost_fn='euclidean' 'logging.wandb_tags=["gt_ref"]'

# python train.py env=Metaworld env.env_reward_type='none' env.task_name='button-press-v2-goal-hidden' reward_model='even_distribution' reward_model.seq_name='rl_expert_20_frames' reward_model.cost_fn='euclidean' 'logging.wandb_tags=["gt_ref"]'

# python train.py env=Metaworld env.env_reward_type='none' env.task_name='button-press-v2-goal-hidden' reward_model='prob_reward' reward_model.seq_name='rl_expert_20_frames' reward_model.cost_fn='euclidean' 'logging.wandb_tags=["gt_ref", "log_prob"]'



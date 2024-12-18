############### BUTTON PRESS ###############

# python train.py env=Metaworld env.env_reward_type='none' env.task_name='button-press-v2-goal-observable' reward_model='dtw' reward_model.seq_name='rl_expert_20_frames' reward_model.cost_fn='euclidean' 'logging.wandb_tags=["gt_ref"]'

# python train.py env=Metaworld env.env_reward_type='none' env.task_name='button-press-v2-goal-observable' env.temporal_encoding=true reward_model='soft_dtw_plus' reward_model.seq_name='rl_expert_20_frames' reward_model.cost_fn='euclidean' 'logging.wandb_tags=["gt_ref"]'

# python train.py env=Metaworld env.env_reward_type='none' env.task_name='button-press-v2-goal-observable' env.temporal_encoding=true reward_model='even_distribution' reward_model.seq_name='rl_expert_20_frames' reward_model.cost_fn='euclidean' 'logging.wandb_tags=["gt_ref", "exp_reward"]'

# python train.py env=Metaworld env.env_reward_type='none' env.task_name='button-press-v2-goal-observable' env.temporal_encoding=true reward_model='prob_ranked' reward_model.seq_name='rl_expert_20_frames' reward_model.cost_fn='euclidean' 'logging.wandb_tags=["gt_ref"]'

# python train.py env=Metaworld env.env_reward_type='sparse' env.task_name='button-press-v2-goal-observable' env.temporal_encoding=true

# python train.py env=Metaworld env.env_reward_type='none' env.task_name='button-press-v2-goal-observable' env.temporal_encoding=true reward_model='prob_reward' reward_model.seq_name='rl_expert_20_frames' reward_model.cost_fn='euclidean' 'logging.wandb_tags=["gt_ref"]'

# python train.py env=Metaworld env.env_reward_type='none' env.task_name='button-press-v2-goal-observable' env.temporal_encoding=true reward_model='log_prob_reward' reward_model.seq_name='rl_expert_20_frames' reward_model.cost_fn='euclidean' 'logging.wandb_tags=["gt_ref", "no_exp"]'

# python train.py env=Metaworld env.env_reward_type='none' env.task_name='button-press-v2-goal-observable' env.temporal_encoding=true reward_model='prob_reward' reward_model.seq_name='rl_expert_20_frames' reward_model.cost_fn='euclidean' 'logging.wandb_tags=["gt_ref", "no_exp"]'

############### DOOR CLOSE ###############

# python train.py env=Metaworld env.env_reward_type='none' env.task_name='door-close-v2-goal-observable' env.temporal_encoding=true reward_model='soft_dtw_plus' reward_model.seq_name='rl_expert_20_frames' reward_model.cost_fn='euclidean' 'logging.wandb_tags=["gt_ref"]'

# python train.py env=Metaworld env.env_reward_type='none' env.task_name='door-close-v2-goal-observable' env.temporal_encoding=true reward_model='prob_reward' reward_model.seq_name='rl_expert_20_frames' reward_model.cost_fn='euclidean' 'logging.wandb_tags=["gt_ref"]' reward_model.max_cost=5

# python train.py env=Metaworld env.env_reward_type='none' env.task_name='door-close-v2-goal-observable' env.temporal_encoding=true reward_model='even_distribution' reward_model.seq_name='rl_expert_20_frames' reward_model.cost_fn='euclidean' 'logging.wandb_tags=["gt_ref", "exp_reward"]'

# python train.py env=Metaworld env.env_reward_type='none' env.task_name='door-close-v2-goal-observable' env.temporal_encoding=true reward_model='log_prob_reward' reward_model.tau=5 reward_model.seq_name='rl_expert_20_frames' reward_model.cost_fn='euclidean' 'logging.wandb_tags=["gt_ref", "no_exp", "tau_5"]'

# python train.py env=Metaworld env.env_reward_type='none' env.task_name='door-close-v2-goal-observable' env.temporal_encoding=true reward_model='coverage' reward_model.tau=1 reward_model.seq_name='rl_expert_20_frames' reward_model.cost_fn='euclidean' 'logging.wandb_tags=["gt_ref", "no_exp", "tau_1"]'


# python train.py env=Metaworld env.env_reward_type='none' env.task_name='door-close-v2-goal-observable' env.temporal_encoding=true reward_model='prob_reward' reward_model.seq_name='rl_expert_20_frames' reward_model.cost_fn='euclidean' 'logging.wandb_tags=["gt_ref", "no_exp"]'



################# DOOR CLOSE VISUAL ##############

# python train.py env=Metaworld env.env_reward_type='none' env.task_name='door-close-v2-goal-observable' env.temporal_encoding=true reward_model='temporal_ot' visual_encoder='resnet50' reward_model.seq_name='rl_expert_20_frames' reward_model.cost_fn='cosine' logging.wandb_mode='online' 

# python train.py env=Metaworld env.env_reward_type='none' env.task_name='door-close-v2-goal-observable' env.temporal_encoding=true reward_model='final_frame' visual_encoder='resnet50' reward_model.seq_name='rl_expert_20_frames' reward_model.cost_fn='cosine' logging.wandb_mode='online' 

# python train.py env=Metaworld env.env_reward_type='none' env.task_name='door-close-v2-goal-observable' env.temporal_encoding=true reward_model='log_prob_reward' visual_encoder='resnet50' reward_model.tau=1 reward_model.seq_name='rl_expert_20_frames' reward_model.cost_fn='cosine' logging.wandb_mode='online' 'logging.wandb_tags=[ "no_exp", "tau_1"]' 

# python train.py env=Metaworld env.env_reward_type='none' env.task_name='door-close-v2-goal-observable' env.temporal_encoding=true reward_model='even_distribution' visual_encoder='resnet50' +reward_model.tau=0.1 reward_model.seq_name='rl_expert_20_frames' reward_model.cost_fn='cosine' logging.wandb_mode='online' 'logging.wandb_tags=[ "no_exp", "tau_01"]' 

# python train.py env=Metaworld env.env_reward_type='none' env.task_name='door-close-v2-goal-observable' env.temporal_encoding=true reward_model='soft_dtw_plus' visual_encoder='resnet50' reward_model.seq_name='rl_expert_20_frames' reward_model.cost_fn='cosine' logging.wandb_mode='online' 

# python train.py env=Metaworld env.env_reward_type='none' env.task_name='door-close-v2-goal-observable' env.temporal_encoding=true reward_model='even_distribution' visual_encoder='resnet50' reward_model.seq_name='rl_expert_20_frames' reward_model.cost_fn='cosine' logging.wandb_mode='online' 

# python train.py env=Metaworld env.env_reward_type='none' env.task_name='door-close-v2-goal-observable' env.temporal_encoding=true reward_model='coverage' visual_encoder='resnet50' reward_model.seq_name='rl_expert_20_frames' reward_model.cost_fn='cosine' logging.wandb_mode='online' 

# python train.py env=Metaworld env.env_reward_type='none' env.task_name='door-close-v2-goal-observable' env.temporal_encoding=true reward_model='prob_reward' visual_encoder='resnet50' reward_model.tau=1 reward_model.seq_name='rl_expert_20_frames' reward_model.cost_fn='cosine' logging.wandb_mode='online' 'logging.wandb_tags=[ "no_exp", "tau_1"]' 

################# BUTTON PRESS VISUAL ##############

# python train.py env=Metaworld env.env_reward_type='none' env.task_name='button-press-v2-goal-observable' env.temporal_encoding=true reward_model='log_prob_reward' visual_encoder='resnet50' reward_model.tau=1 reward_model.seq_name='rl_expert_20_frames' reward_model.cost_fn='cosine' logging.wandb_mode='online' 'logging.wandb_tags=["no_exp", "tau_1"]' 

# python train.py env=Metaworld env.env_reward_type='none' env.task_name='button-press-v2-goal-observable' env.temporal_encoding=true reward_model='temporal_ot' visual_encoder='resnet50' reward_model.seq_name='rl_expert_20_frames' reward_model.cost_fn='cosine' logging.wandb_mode='online' 

# python train.py env=Metaworld env.env_reward_type='none' env.task_name='button-press-v2-goal-observable' env.temporal_encoding=true reward_model='soft_dtw_plus' visual_encoder='resnet50' reward_model.seq_name='rl_expert_20_frames' reward_model.cost_fn='cosine' logging.wandb_mode='online' 

# python train.py env=Metaworld env.env_reward_type='none' env.task_name='button-press-v2-goal-observable' env.temporal_encoding=true reward_model='even_distribution' visual_encoder='resnet50' reward_model.seq_name='rl_expert_20_frames' reward_model.cost_fn='cosine' logging.wandb_mode='online' 
python train.py env=Metaworld env.env_reward_type='none' env.task_name='bin-picking-v2-goal-observable' env.temporal_encoding=true reward_model='temporal_ot' visual_encoder='resnet50' reward_model.cost_fn='diagonal_cosine' logging.wandb_mode='online' reward_model.tau=0.1

# python train.py env=Metaworld env.env_reward_type='none' env.task_name='button-press-v2-goal-observable' env.temporal_encoding=true reward_model='log_prob_reward' visual_encoder='resnet50' reward_model.seq_name='rl_expert_20_frames' reward_model.cost_fn='diagonal_cosine' logging.wandb_mode='online' reward_model.tau=1

# python train.py env=Metaworld env.env_reward_type='none' env.task_name='button-press-v2-goal-observable' env.temporal_encoding=true reward_model='coverage' visual_encoder='resnet50' reward_model.seq_name='rl_expert_20_frames' reward_model.cost_fn='smoothed_cosine' +reward_model.context_window=8 logging.wandb_mode='online' 


# python train.py env=Metaworld env.env_reward_type='none' env.task_name='button-press-v2-goal-observable' env.temporal_encoding=true reward_model='prob_reward' visual_encoder='resnet50' reward_model.tau=1 reward_model.seq_name='rl_expert_20_frames' reward_model.cost_fn='cosine' logging.wandb_mode='online' 'logging.wandb_tags=[ "no_exp", "tau_1"]' 


################# WINDOW OPEN VISUAL ##############

# python train.py env=Metaworld env.env_reward_type='none' env.task_name='window-open-v2-goal-observable' env.temporal_encoding=true reward_model='log_prob_reward' visual_encoder='resnet50' reward_model.tau=1 reward_model.seq_name='rl_expert_20_frames' reward_model.cost_fn='cosine' logging.wandb_mode='online' 'logging.wandb_tags=["no_exp", "tau_1"]' 

# python train.py env=Metaworld env.env_reward_type='none' env.task_name='window-open-v2-goal-observable' env.temporal_encoding=true reward_model='temporal_ot' visual_encoder='resnet50' reward_model.seq_name='rl_expert_20_frames' reward_model.cost_fn='cosine' logging.wandb_mode='online' 

# python train.py env=Metaworld env.env_reward_type='none' env.task_name='window-open-v2-goal-observable' env.temporal_encoding=true reward_model='coverage' visual_encoder='resnet50' reward_model.tau=1 reward_model.seq_name='rl_expert_20_frames' reward_model.cost_fn='cosine' logging.wandb_mode='online' 'logging.wandb_tags=["no_exp", "tau_1"]' 

# python train.py env=Metaworld env.env_reward_type='none' env.task_name='window-open-v2-goal-observable' env.temporal_encoding=true reward_model='soft_dtw_plus' visual_encoder='resnet50' reward_model.seq_name='rl_expert_20_frames' reward_model.cost_fn='cosine' logging.wandb_mode='online' 

# python train.py env=Metaworld env.env_reward_type='none' env.task_name='window-open-v2-goal-observable' env.temporal_encoding=true reward_model='even_distribution' visual_encoder='resnet50' reward_model.seq_name='rl_expert_20_frames' reward_model.cost_fn='cosine' logging.wandb_mode='online' 



# python train.py env=Metaworld env.env_reward_type='none' env.task_name='window-open-v2-goal-observable' env.temporal_encoding=true reward_model='prob_reward' visual_encoder='resnet50' reward_model.tau=1 reward_model.seq_name='rl_expert_20_frames' reward_model.cost_fn='cosine' logging.wandb_mode='online' 'logging.wandb_tags=[ "no_exp", "tau_1"]' 


################# DOOR OPEN VISUAL ##############

# python train.py env=Metaworld env.env_reward_type='none' env.task_name='door-open-v2-goal-observable' env.temporal_encoding=true reward_model='log_prob_reward' visual_encoder='resnet50' reward_model.tau=1 reward_model.seq_name='rl_expert_20_frames' reward_model.cost_fn='cosine' logging.wandb_mode='online' 'logging.wandb_tags=["no_exp", "tau_1"]' 

# python train.py env=Metaworld env.env_reward_type='none' env.task_name='door-open-v2-goal-observable' env.temporal_encoding=true reward_model='temporal_ot' visual_encoder='resnet50' reward_model.seq_name='rl_expert_20_frames' reward_model.cost_fn='cosine' logging.wandb_mode='online' 

# python train.py env=Metaworld env.env_reward_type='none' env.task_name='door-open-v2-goal-observable' env.temporal_encoding=true reward_model='coverage' visual_encoder='resnet50' reward_model.tau=1 reward_model.seq_name='rl_expert_20_frames' reward_model.cost_fn='cosine' logging.wandb_mode='online' 'logging.wandb_tags=["no_exp", "tau_1"]' 

# python train.py env=Metaworld env.env_reward_type='none' env.task_name='door-open-v2-goal-observable' env.temporal_encoding=true reward_model='soft_dtw_plus' visual_encoder='resnet50' reward_model.seq_name='rl_expert_20_frames' reward_model.cost_fn='cosine' logging.wandb_mode='online' 

# python train.py env=Metaworld env.env_reward_type='none' env.task_name='door-open-v2-goal-observable' env.temporal_encoding=true reward_model='even_distribution' visual_encoder='resnet50' reward_model.seq_name='rl_expert_20_frames' reward_model.cost_fn='cosine' logging.wandb_mode='online' 


# WARNING!!!!!!!!!!!
# reward_batch_size batch size per synchronous inference step.
# reward_batch_size must be able to divisible by
#      (0.8 * reward_batch_size) * (n_workers - 1)
#       so that it can be shared among workers,
# and must be 
#       a divisor of n_envs * episode_length 
#       so that all batches can be of the same size
####################################### Ours (DINO-based)
python train.py \
    -algo crossq \
    -env HumanoidSpawnedUpCustom \
    -reward_type simple_remain_standing\
    -n_envs 8 \
    -seed 9 \
    -n_workers 2\
    -rank0_batch_size_pct 0\
    -reward_model_name 'dinov2_vitl14_reg'\
    -reward_batch_size 30\
    -reward_config './configs/dino_splits_config_mujoco.yml'\
    -wandb_mode 'online' \
    -total_timesteps=1000000 \
    -eval_freq=100000 \
    -model_save_freq=50000 \
    -video_save_freq=10000 \
    -episode_length=120
    #-model_checkpoint final_model \
    #-model_base_path /share/portal/hw575/CrossQ/train_logs/crossq_HumanoidSpawnedUpCustom_r=kneeling_s=9_2024-05-19_21-43-48__d7adbd23/checkpoint

######## Public G2 A6000s (must decrease batch size for some reason)
# python train.py \
#     -algo crossq \
#     -env HumanoidSpawnedUpCustom \
#     -reward_type simple_remain_standing\
#     -n_envs 8 \
#     -seed 9 \
#     -n_workers 2\
#     -rank0_batch_size_pct 0.2\
#     -reward_model_name 'dinov2_vitl14_reg'\
#     -reward_batch_size 60\
#     -reward_config './configs/dino_reward_config.yml'\
#     -wandb_mode 'online' \
#     -total_timesteps=4000000 \
#     -eval_freq=100000 \
#     -model_save_freq=100000 \
#     -video_save_freq=10000 \
#     -episode_length=240 \
# ----------- Using 1 GPU (8 CPU)
# python train.py \
#     -algo crossq \
#     -env HumanoidSpawnedUpCustom \
#     -reward_type simple_remain_standing\
#     -n_envs 8 \
#     -seed 9 \
#     -n_workers 1\
#     -rank0_batch_size_pct 1.0\
#     -reward_model_name 'dinov2_vitl14_reg'\
#     -reward_batch_size 32\
#     -reward_config './configs/dino_reward_config.yml'\
#     -wandb_mode 'online' \
#     -total_timesteps=1000000 \
#     -eval_freq=100000 \
#     -model_save_freq=100000 \
#     -video_save_freq=10000 \
#     -episode_length=256 \
#     --no-distributed
######################################## CLIP
    - preference_data/splits_f_target_frame231.png
# python train.py \
#     -algo crossq \
#     -env HumanoidSpawnedUpCustom \
#     -reward_type simple_remain_standing\
#     -n_envs 8 \
#     -seed 9 \
#     -n_workers 2\
#     -reward_model_name 'ViT-g-14/laion2b_s34b_b88k'\
#     -reward_batch_size 240\
#     -reward_config './configs/clip_reward_config.yml'\
#     -wandb_mode 'online' \
#     -total_timesteps=1000000 \
#     -eval_freq=100000 \
#     -model_save_freq=100000 \
#     -video_save_freq=10000 \
#     -episode_length=240 \

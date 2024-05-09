# CLIP batch size per synchronous inference step.
# Batch size must be divisible by n_workers (GPU count)
# so that it can be shared among workers, and must be a divisor
# of n_envs * episode_length so that all batches can be of the
# same size (no support for variable batch size as of now.)
# python train.py \
#     -algo crossq \
#     -env HumanoidSpawnedUp \
#     -reward_type remain_standing\
#     -n_envs 8 \
#     -seed 9 \
#     -n_workers 2\
#     -reward_model_name 'dinov2_vits14_reg'\
#     -reward_batch_size 16\
#     -reward_config './configs/dino_reward_config.yml'\
#     -wandb_mode 'online' \
#     -total_timesteps=1000000 \
#     -eval_freq=100000 \
#     -model_save_freq=100000 \
#     -video_save_freq=10000 \
#     -episode_length=240 \
######################################## CLIP
python train.py \
    -algo crossq \
    -env HumanoidSpawnedUpCustom \
    -reward_type best_standing_up\
    -n_envs 8 \
    -seed 9 \
    -n_workers 2\
    -reward_model_name 'ViT-g-14/laion2b_s34b_b88k'\
    -reward_batch_size 240\
    -reward_config './configs/clip_reward_config.yml'\
    -wandb_mode 'online' \
    -total_timesteps=1000000 \
    -eval_freq=100000 \
    -model_save_freq=100000 \
    -video_save_freq=10000 \
    -episode_length=240 \
# python train.py \
#     -algo crossq \
#     -env HumanoidStandupCurriculum \
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
#     -video_save_freq=25000 \
#     -episode_length=240 \

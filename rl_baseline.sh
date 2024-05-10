################################################################### CrossQ
python train.py \
    -algo crossq \
    -env HumanoidSpawnedUpCustom \
    -reward_type simple_remain_standing \
    -n_envs 8 \
    -seed 9 \
    -wandb_mode 'online' \
    -total_timesteps=1000000 \
    -model_save_freq=100000 \
    -video_save_freq=10000 \
    -episode_length=240 \
################################################################### SAC
# python train.py \
#     -algo sac \
#     -env HumanoidSpawnedUpCustom \
#     -reward_type simple_remain_standing \
#     -n_envs 8 \
#     -seed 9 \
#     -wandb_mode 'online' \
#     -total_timesteps=1000000 \
#     -model_save_freq=100000 \
#     -video_save_freq=10000 \
#     -episode_length=240 \

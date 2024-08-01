################################################################### CrossQ
python train.py \
    -algo crossq \
    -env HumanoidSpawnedUpCustom \
    -reward_type both_arms_out_goal_only_euclidean \
    -n_envs 8 \
    -seed 9 \
    -wandb_mode 'online' \
    -total_timesteps=1000000 \
    -model_save_freq=100000 \
    -video_save_freq=10000 \
    -episode_length=120 \
    -run_notes '1*pose_r + 0*uph_r - 1*ctrl_c'
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

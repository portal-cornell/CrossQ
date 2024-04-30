# python train.py \
#     -algo crossq \
#     -env HumanoidStandupCurriculum \
#     -reward_type tassa_mpc_torso_bottom_up\
#     -num_envs 8 \
#     -seed 9 \
#     -wandb_mode 'online' \
#     -total_timesteps=2000000 \
#     -model_save_freq=200000 \
#     -video_save_freq=100000 \
#     -episode_length=240 \
python train.py \
    -algo crossq \
    -env HumanoidStandupCurriculum \
    -reward_type tassa_imp_circle_dist\
    -num_envs 8 \
    -seed 9 \
    -wandb_mode 'online' \
    -total_timesteps=5000000 \
    -model_save_freq=500000 \
    -video_save_freq=250000 \
    -episode_length=240 \
# python train.py \
#     -algo crossq \
#     -env HumanoidStandupCurriculum \
#     -reward_type tassa_mpc_torso_bottom_up\
#     -num_envs 8 \
#     -seed 9 \
#     -wandb_mode 'online' \
#     -total_timesteps=10000000 \
#     -model_save_freq=1000000 \
#     -video_save_freq=500000 \
#     -episode_length=240 \

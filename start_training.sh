python train.py \
    -algo crossq \
    -env HumanoidStandupCurriculum \
    -num_envs 8 \
    -seed 9 \
    -wandb_mode 'online' \
    -total_timesteps=1000000 \
    -model_save_freq=100000 \
    -video_save_freq=50000 \
    -episode_length=240 \

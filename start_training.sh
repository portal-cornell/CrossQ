python train.py \
    -algo crossq \
    -env HumanoidStandup-v4 \
    -num_envs 8 \
    -seed 9 \
    -wandb_mode 'online' \
    -total_timesteps=2000000 \
    -model_save_freq=200000 \
    -video_save_freq=100000 \
    -episode_length=240 \

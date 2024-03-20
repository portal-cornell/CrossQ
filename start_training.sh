python train.py \
    -algo crossq \
    -env HumanoidStandup-v4 \
    -seed 9 \
    -wandb_mode 'online' \
    -total_timesteps=5000000 \
    -model_save_freq=100000
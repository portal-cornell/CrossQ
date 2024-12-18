#!/bin/bash

# Job Parameters
PARTITION="gpu" # "portal-interactive"
CPUS=8
GPUS=1
MEMORY=35GB
TIME="4:00:00"

# Training Parameters
TAU=(0.1 0.5 1)
DISCOUNT_FACTOR=(0.9 0.99)

ENV="Metaworld"
ENV_REWARD_TYPE="none"
TASK_NAME="door-open-v2-goal-observable"
TEMPORAL_ENCODING="true"
REWARD_MODEL="coverage"
VISUAL_ENCODER="resnet50"
COST_FN="diagonal_cosine"
WANDB_MODE="online"
LOG_FREQ=50000 # Log every LOG_FREQ steps

for tau in "${TAU[@]}"; do
    for discount_factor in "${DISCOUNT_FACTOR[@]}"; do
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=train_tau_${tau}
#SBATCH --partition=${PARTITION}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --gres=gpu:${GPUS}
#SBATCH --mem=${MEMORY}
#SBATCH --time=${TIME}
#SBATCH --output=dump/train_tau_${tau}_%j.out
#SBATCH --error=dump/train_tau_${tau}_%j.err

python train.py \
    env=${ENV} \
    env.env_reward_type=${ENV_REWARD_TYPE} \
    env.task_name=${TASK_NAME} \
    env.temporal_encoding=${TEMPORAL_ENCODING} \
    reward_model=${REWARD_MODEL} \
    visual_encoder=${VISUAL_ENCODER} \
    reward_model.cost_fn=${COST_FN} \
    logging.wandb_mode=${WANDB_MODE} \
    logging.video_save_freq=${LOG_FREQ} \
    reward_model.tau=${tau} \
    rl_algo.discount_factor=${discount_factor}
EOF
sleep 1.1 # make sure the new wandb folder is different (seconds is the identifier)
    done
done
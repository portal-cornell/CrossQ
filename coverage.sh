#!/bin/bash

# Job Parameters
PARTITION="gpu" # "portal-interactive"
CPUS=8
GPUS=1
MEMORY=35GB
TIME="4:00:00"

# Training Parameters
TAU=1
DISCOUNT_FACTOR=0.99
ENV="Metaworld"
ENV_REWARD_TYPE="none"
TASK_NAME=("button-press-v2-goal-observable" "door-close-v2-goal-observable" "door-lock-v2-goal-observable" "hammer-v2-goal-observable" "box-close-v2-goal-observable" "assembly-v2-goal-observable")
TEMPORAL_ENCODING="true"
REWARD_MODEL="log_prob_reward"
VISUAL_ENCODER="resnet50"
COST_FN="diagonal_cosine"
WANDB_MODE="online"
LOG_FREQ=50000 # Log every LOG_FREQ steps
SEED=42

for task_name in "${TASK_NAME[@]}"; do
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
    env.task_name=${task_name} \
    env.temporal_encoding=${TEMPORAL_ENCODING} \
    reward_model=${REWARD_MODEL} \
    visual_encoder=${VISUAL_ENCODER} \
    reward_model.cost_fn=${COST_FN} \
    logging.wandb_mode=${WANDB_MODE} \
    logging.video_save_freq=${LOG_FREQ} \
    reward_model.tau=${TAU} \
    rl_algo.discount_factor=${DISCOUNT_FACTOR} \
    seed=${SEED}
EOF
sleep 1.1 # make sure the new wandb folder is different (seconds is the identifier)
done

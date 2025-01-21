#!/bin/bash

# Job Parameters
PARTITION="gpu"
CPUS=8
GPUS=1
MEMORY=35GB
TIME="10:00:00"
TASK_NAME="both_arms_out" # Example: "right_arm_extend_wave_higher"
REWARD_FN="coverage" # Example: "temporal_ot"
SCALE_BEFORE_MATCHING=("False") # for coverage
SCALE_BY_FIRST_ROLLOUT=("False")
TAU=(1 10) # for coverage
WANDB_MODE="online"

for scale_before in "${SCALE_BEFORE_MATCHING[@]}"; do
    for scale_by_first in "${SCALE_BY_FIRST_ROLLOUT[@]}"; do
        for tau in "${TAU[@]}"; do
            sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=train-${TASK_NAME}-${REWARD_FN}
#SBATCH --partition=${PARTITION}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --gres=gpu:${GPUS}
#SBATCH --mem=${MEMORY}
#SBATCH --time=${TIME}
#SBATCH --output=dump/train_%j.out
#SBATCH --error=dump/train_%j.err

# Capture the Slurm job ID
job_id=\$SLURM_JOB_ID

echo "Running training for job ID: \$job_id"
python -m train \
    "visual_reward_model=joint_pred_resnet" \
    "matching_reward_model=${REWARD_FN}" \
    "compute.n_gpu_workers=1" \
    "env.task_name=${TASK_NAME}" \
    "env.reward_type=basic_r_geom_xpos" \
    "visual_reward_model.scale_uncertainty_before_matching=${scale_before}" \
    "matching_reward_model.scale_by_first_rollout=${scale_by_first}" \
    "matching_reward_model.tau=${tau}" \
    "visual_reward_model.use_image_for_ref=True" \
    "logging.wandb_mode=${WANDB_MODE}" \
    "++matching_reward_model.tau=${tau}" \
    "matching_reward_model.seq_name=intermediate_10_frames"
EOF
            sleep 1.1 # Ensure a unique timestamp for each run
        done
    done
done

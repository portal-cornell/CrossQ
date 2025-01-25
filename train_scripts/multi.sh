#!/bin/bash

# Job Parameters
PARTITION="gpu"
CPUS=8
GPUS=1
MEMORY=35GB
TIME="10:00:00"
TASK_NAME=("right_arm_out") # "right_arm_out" "left_arm_out") #"both_arms_out") # #("both_arms_down") #
REWARD_FN=("coverage") #"temporal_ot"
SEED=("r" "r")
SCALE_BEFORE_MATCHING="False" # for coverage
TAU=1 # for coverage
WANDB_MODE="online"

for task_name_i in "${TASK_NAME[@]}"; do
    for reward_fn_i in "${REWARD_FN[@]}"; do
        for seed_i in "${SEED[@]}"; do
            sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=train-${task_name_i}-${reward_fn_i}
#SBATCH --partition=${PARTITION}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --gres=gpu:${GPUS}
#SBATCH --mem=${MEMORY}
#SBATCH --time=${TIME}
#SBATCH --output=dump/train_%j.out
#SBATCH --error=dump/train_%j.err

# Capture the Slurm job ID
job_id=\$SLURM_JOB_ID

echo "Running training for task: ${task_name_i}, job ID: \${job_id}"
python -m train \
    "visual_reward_model=joint_pred_resnet" \
    "seed=${seed_i}"\
    "matching_reward_model=${reward_fn_i}" \
    "compute.n_gpu_workers=1" \
    "env.task_name=${task_name_i}" \
    "env.reward_type=basic_r_geom_xpos" \
    "visual_reward_model.scale_uncertainty_before_matching=${SCALE_BEFORE_MATCHING}" \
    "visual_reward_model.use_image_for_ref=True" \
    "logging.wandb_mode=${WANDB_MODE}" \
    "++matching_reward_model.tau=${TAU}" \
    "matching_reward_model.seq_name=intermediate_10_frames"
EOF
            sleep 1.1 # Ensure a unique timestamp for each run
        done
    done
done

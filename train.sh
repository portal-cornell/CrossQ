#!/bin/bash

# Job Parameters
PARTITION="gpu"
CPUS=8
GPUS=1
MEMORY=35GB
TIME="10:00:00"
TASK_NAME=("left_arm_extend_wave_higher") # "right_arm_out" "left_arm_out") #"both_arms_out") # #("both_arms_down") #
REWARD_FN=("temporal_ot") #("dtw" "optimal_transport" "temporal_ot") #"temporal_ot"
SEED=(2675) 
SCALE_BEFORE_MATCHING="False"
TAU=1 # for coverage
MASK_K=1
WANDB_MODE="online"

for task_name_i in "${TASK_NAME[@]}"; do
    for reward_fn_i in "${REWARD_FN[@]}"; do
        for seed_i in "${SEED[@]}"; do
            echo "Running training for task: ${task_name_i}"

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
                "++matching_reward_model.mask_k=${MASK_K}" \
                "matching_reward_model.seq_name=intermediate_10_frames"

            sleep 1.1 # Ensure a unique timestamp for each run
        done
    done
done

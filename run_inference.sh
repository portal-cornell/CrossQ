python inference.py \
    -env HumanoidStandupCurriculum \
    -reward_type stage1_v1\
    -seed 9 \
    -model_checkpoint final_model \
    -model_base_path /share/portal/hw575/CrossQ/train_logs/CrossQ_HumanoidStandupCurriculum_name=HumanoidStandupCurriculum_s=9_2024-04-21_23-28-39__127d5bd0/checkpoint \
    -episode_length 240

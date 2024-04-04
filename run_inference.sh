python inference.py \
    -env HumanoidStandupCurriculum \
    -reward_type stage1_v1\
    -seed 9 \
    -model_checkpoint final_model \
    -model_base_path /share/portal/hw575/CrossQ/train_logs/CrossQ_HumanoidStandupCurriculum_name=HumanoidStandupCurriculum_stage=0_r=stage1-v1_s=9_2024-03-28_01-56-24__74467695/checkpoint \
    -episode_length 240

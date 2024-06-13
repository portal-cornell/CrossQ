# python inference.py \
#     -env HumanoidSpawnedUpCustom \
#     -reward_type best_standing_up\
#     -seed 9 \
#     -reward_model_name 'dinov2_vitl14_reg'\
#     -reward_batch_size 16\
#     -reward_config './configs/dino_reward_config.yml'\
#     -model_checkpoint final_model \
#     -model_base_path /share/portal/hw575/CrossQ/train_logs/crossq_HumanoidSpawnedUpCustom_rm=dino_r=best_standing_up_s=9_2024-05-08_23-33-09__ced2cfc2/checkpoint/ \
#     -episode_length 240
python inference.py \
    -env 'HumanoidSpawnedUpCustom' \
    -reward_type 'kneeling'\
    -seed 9 \
    -reward_model_name 'dinov2_vitl14_reg'\
    -reward_batch_size 4\
    -reward_config './configs/dino_kneeling_config.yml'\
    -model_checkpoint 'model_50000_steps' \
    -model_base_path 'train_logs/crossq_HumanoidSpawnedUpCustom_rm=dino_r=simple_remain_standing_s=9_2024-06-10_19-16-05__7167a013/checkpoint' \
    -episode_length 120
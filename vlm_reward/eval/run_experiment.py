from vlm_reward.eval.stage2_reward_eval import main 
import subprocess
import os

def override_tasks():

    eval_data_configs = ['crossq_kneeling', 'crossq_splits']
    target_images = ['/share/portal/wph52/CrossQ/vlm_reward/eval/target_images/humanoid_kneeling_ref.png',
                    '/share/portal/wph52/CrossQ/vlm_reward/eval/target_images/splits_f_frame_0300.png']


    for eval_cfg, trg in zip(eval_data_configs, target_images):
        command = [
            "python",
            "stage2_reward_eval.py",
            f"eval_data={eval_cfg}",
            f"reward_model.pos_image_path={[trg]}"
        ]
        result = subprocess.run(command)

def override_reward_model():
    models = ["dino_pooled"]
    
    for model in models:
        command = [
            "python",
            "stage2_reward_eval.py",
            f"eval_data=crossq_splits",
            f"reward_model={model}",
            f"reward_model.pos_image_path={['/share/portal/wph52/CrossQ/vlm_reward/eval/target_images/splits_f_frame_0300.png']}"
        ]
        result = subprocess.run(command)

def override_tasks_and_model(target_images, target_image_type):

    eval_data_configs = ['crossq_arms_out', 'crossq_kneeling', 'crossq_splits']

    models = ["lpips", "dreamsim", "dino_pooled", "dino_patch_wasserstein"]    
    
    for eval_cfg, trg in zip(eval_data_configs, target_images):
        trg_name = os.path.basename(trg).split('.')[0]
        for model in models:
            command = [
                "python",
                "stage2_reward_eval.py",
                f"eval_data={eval_cfg}",
                f"reward_model={model}",
                f"reward_model.pos_image_path={[trg]}",
                f"run_notes={trg_name}",
                f"run_type={target_image_type}"
            ]
            result = subprocess.run(command)

def mujoco_goal_image_exp():
    # mujoco goal images
    target_images = ['/share/portal/wph52/CrossQ/create_demo/demos/both-arms-out.png',
                    '/share/portal/wph52/CrossQ/vlm_reward/eval/target_images/humanoid_kneeling_ref.png',
                    '/share/portal/wph52/CrossQ/vlm_reward/eval/target_images/splits_f_frame_0300.png']
    run_type = "mujoco-goal"

    override_tasks_and_model(target_images, run_type)


def human_goal_image_exp():
    # human goal images
    target_images = ['/share/portal/wph52/CrossQ/vlm_reward/eval/target_images/human-t-pose.jpeg',
                    '/share/portal/wph52/data/preference_data/humans_selected_images/kneeling/anne_kneeling_front_final.png',
                    '/share/portal/wph52/data/preference_data/humans_selected_images/splits/internet_split_final.png']
    run_type = "human-goal"

    override_tasks_and_model(target_images, run_type)

if __name__=="__main__":
    mujoco_goal_image_exp()
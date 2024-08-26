from vlm_reward.eval.stage2_reward_eval import main 
import subprocess

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

def override_tasks_and_model():

    eval_data_configs = ['crossq_arms_out', 'crossq_kneeling', 'crossq_splits']
    target_images = ['/share/portal/wph52/CrossQ/create_demo/demos/both-arms-out.png',
                    '/share/portal/wph52/CrossQ/vlm_reward/eval/target_images/humanoid_kneeling_ref.png',
                    '/share/portal/wph52/CrossQ/vlm_reward/eval/target_images/splits_f_frame_0300.png'
                    ]

    models = ["lpips", "dreamsim", "dino_pooled", "dino_patch_wasserstein"]    
    
    for eval_cfg, trg in zip(eval_data_configs, target_images):
        for model in models:
            command = [
                "python",
                "stage2_reward_eval.py",
                f"eval_data={eval_cfg}",
                f"reward_model={model}",
                f"reward_model.pos_image_path={[trg]}"
            ]
            result = subprocess.run(command)

if __name__=="__main__":
    override_tasks_and_model()
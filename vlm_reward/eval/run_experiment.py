from stage2_reward_eval import main 
import subprocess

def override_main():

    eval_data_configs = ['crossq_kneeling', 'crossq_splits']
    target_images = ['/share/portal/wph52/CrossQ/vlm_reward/eval/target_images/humanoid_kneeling_ref.png',
                    '/share/portal/wph52/CrossQ/vlm_reward/eval/target_images/splits_f_frame_0300.png',]

    for eval_cfg, trg in zip(eval_data_configs, target_images):
        command = [
            "python",
            "stage2_reward_eval.py",
            f"eval_data={eval_cfg}",
            f"reward_model.pos_image_path={[trg]}"
        ]

        result = subprocess.run(command)


if __name__=="__main__":
    override_main()
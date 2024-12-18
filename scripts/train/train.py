

import subprocess

def run_train_script(args):

    # Base command
    command = ["python", "train.py"]

    # Convert the dictionary to a list of command-line arguments
    for key, value in args.items():
        command.append(f"+{key}={value}")

    # Run the command
    result = subprocess.run(command, capture_output=True, text=True)

    # Print the outputs (optional)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

def run_tau_experiments():
    args = {
        "env": "Metaworld",
        "env.env_reward_type": "none",
        "env.task_name": "window-open-v2-goal-observable",
        "env.temporal_encoding": "true",
        "reward_model": "log_prob_reward",
        "visual_encoder": "resnet50",
        "reward_model.tau": 1,
        "reward_model.seq_name": "rl_expert_20_frames",
        "reward_model.cost_fn": "cosine",
        "logging.wandb_mode": "online",
        "logging.wandb_tags": '["no_exp", "tau_1"]'
    }

def run_model_experiments():
    args = {
        "env": "Metaworld",
        "env.env_reward_type": "none",
        "env.task_name": "window-open-v2-goal-observable",
        "env.temporal_encoding": "true",
        "reward_model": "log_prob_reward",
        "visual_encoder": "resnet50",
        "reward_model.tau": 1,
        "reward_model.seq_name": "rl_expert_20_frames",
        "reward_model.cost_fn": "cosine",
        "logging.wandb_mode": "online",
        "logging.wandb_tags": '["no_exp", "tau_1"]'
    }


if __name__ == "__main__":
    run_train_script()

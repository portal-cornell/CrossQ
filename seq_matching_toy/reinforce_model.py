import torch
import time
import wandb
import sys

import numpy as np
from stable_baselines3.common.utils import safe_mean
from tqdm import tqdm
from PIL import Image


# A simple MLP policy for REINFORCE
class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 32)
        self.fc2 = torch.nn.Linear(32, output_dim)
        
    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=-1)
    

class REINFORCE(object):
    def __init__(self,
                 env,
                 learning_rate=0.01,
                 ent_coef=0.01,
                 video_save_freq=500):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = MLP(env.observation_space.shape[0], env.action_space.n).to(self.device)

        self.env = env

        self.last_video_save_step = 0
        self.video_save_freq = video_save_freq

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.ent_coef = ent_coef
        
    def learn(self, total_timesteps, progress_bar=True, callback=[]):
        self.start_time = time.time()

        self.num_timesteps = 0

        print(f"Training for {total_timesteps} timesteps, {total_timesteps // self.env.episode_length} episodes")   

        for i in tqdm(range(0, total_timesteps, self.env.episode_length)):
            # Collect rollouts
            #   TODO: hack to assume that the last callback is the seq_matching_callback
            rollout_dict = self.collect_rollouts(self.env, callback[-1])

            log_probs = rollout_dict["log_probs"]
            rewards = torch.tensor(rollout_dict["rewards"]).to(self.device)
            all_log_probs = rollout_dict["all_log_probs"]

            self.num_timesteps += self.env.episode_length

            policy_loss, entropy_loss = self.train(log_probs, rewards, all_log_probs)

            print(f"[{i}] avg reward: {np.mean(rollout_dict['rewards'])}, pol={policy_loss:.2f}, ent={entropy_loss:.2f}, {all_log_probs[-1].data}")

            time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
            wandb.log({"time/iterations": i,
                       "rollout/ep_rew_mean": np.mean(rollout_dict["rewards"]),
                       "rollout/ep_rew_sum": np.sum(rollout_dict["rewards"]),
                       "time/time_elapsed": int(time_elapsed),
                       "train/policy_loss": policy_loss,
                       "train/entropy_loss": entropy_loss,
                       }, step=self.num_timesteps)
            
            if self.last_video_save_step == 0 or (self.num_timesteps - self.last_video_save_step) >= self.video_save_freq:
                self.evaluate_policy(self.env, 
                                     seq_matching_callback=callback[-1], 
                                     video_save_callback=callback[-2])
                
                self.last_video_save_step = self.num_timesteps

    def train(self, log_probs, rewards, all_log_probs):
        log_probs = torch.stack(log_probs)
        all_log_probs = torch.stack(all_log_probs)
        policy_loss = -torch.mean(log_probs * rewards)
        # Calculate the entropy of the policy with equation: H(p) = -sum(p(x) * log(p(x)))
        entropy_loss = torch.mean(torch.sum(torch.exp(all_log_probs) * all_log_probs, dim=1))
        loss = policy_loss + self.ent_coef * entropy_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return policy_loss.item(), entropy_loss.item()
    
    def collect_rollouts(self, env, seq_matching_callback, deterministic=False, save_video=False):
        state, _ = env.reset()
        done = False
        log_probs = []
        all_log_probs = []
        rewards = []
        states = []
        actions = []
        infos = []

        raw_screens = []
        screens = []
        
        while not done:
            # Select action
            state = torch.tensor(state, dtype=torch.float32).reshape(1, -1).to(self.device)
            probs = self.policy(state)
            if deterministic:
                action = torch.argmax(probs).item()
            else:
                action = torch.multinomial(probs, 1).item()  # Same as categorical distribution
            log_prob = torch.log(probs[:, action])
            all_log_probs.append(torch.log(probs))

            # Take step
            next_state, reward, done, _, info = env.step(action)

            if save_video:
                screen = env.render()
                raw_screens.append(Image.fromarray(np.uint8(screen)))
                screens.append(Image.fromarray(np.uint8(screen)))
                infos.append(info)

            rewards.append(reward)
            log_probs.append(log_prob)
            states.append(state.cpu().numpy())
            actions.append(action)
            state = next_state

        states = np.concatenate(states)

        seq_matching_rewards = seq_matching_callback.on_rollout_end_no_buffer(np.array(states), np.array(actions), np.array(rewards))

        return {
            "log_probs": log_probs,
            "all_log_probs": all_log_probs,
            "raw_rewards": rewards,
            "rewards": seq_matching_rewards,
            "states": states,
            "actions": actions,
            "raw_screens": raw_screens,
            "screens": screens,
            "info": infos
        }
    
    def evaluate_policy(self, env, seq_matching_callback, video_save_callback):
        rollout_dict = self.collect_rollouts(self.env, seq_matching_callback, deterministic=True, save_video=True)

        video_save_callback._on_step_no_buffer(
            raw_screens=rollout_dict["raw_screens"],
            screens=rollout_dict["screens"],
            states=rollout_dict["states"],
            actions=rollout_dict["actions"],
            rewards=rollout_dict["raw_rewards"],
            infos=rollout_dict["info"],
            num_timesteps=self.num_timesteps
        )

    
    def save(self, path):
        pass
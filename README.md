# Setup
1. Install all the necessary packages
```
conda create -n <your env name> python=3.11.5
```

For the path that you need to export, (`export PATH=/home/hw575/.conda/envs/crossq/bin:$PATH` is an example). You can verify by making sure that `which pip` points to the pip in your conda environment
```
conda activate <your env name>
export PATH=<path to your conda env>:$PATH
```


```
conda install -c nvidia cuda-nvcc=12.4.99

pip install -e .

pip install --upgrade "jax[cuda12_pip]"==0.4.23 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

pip install -r additional_requirements.txt

pip install torch==2.1.0 torchvision==0.16.0

pip install nvidia-cublas-cu12==12.4.2.65 nvidia-cuda-cupti-cu12==12.4.99 nvidia-cuda-nvrtc-cu12==12.4.99 nvidia-cuda-runtime-cu12==12.4.99 nvidia-cudnn-cu12==8.9.7.29 nvidia-cufft-cu12==11.2.0.44 nvidia-cusolver-cu12==11.6.0.99 nvidia-cusparse-cu12==12.3.0.142 nvidia-nccl-cu12==2.20.5
```

2. run `git submodule update --init --recursive`

You can verify that if you `cd sbx/vlm_reward/reward_models/language_irl` and `git status`, it should be on the `for-crossq-env` branch.

You should also verify that in the language_irl folder `SemnaticGuidedHumanMatting` is properly initialized (there are files there)

3. Replace these with path to your local file
- In `callbacks.py` and `inference.py`, we need arial.ttf font to plot the reward number on the rollout video. You can download it online.
- In `constants.py`, point the WANDB_DIR to the path in your CrossQ folder
- In `configs/clip_reward_config.yml` and `configs/dino_reward_config.yml`, make sure the clip cached_dir, human_seg_model_path, and reference image path are pointed to your folders.


## pip warning
Don't worry if you see the following:
> ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
torch 2.1.2 requires nvidia-cublas-cu12==12.1.3.1; platform_system == "Linux" and platform_machine == "x86_64", but you have nvidia-cublas-cu12 12.4.5.8 which is incompatible.
torch 2.1.2 requires nvidia-cuda-cupti-cu12==12.1.105; platform_system == "Linux" and platform_machine == "x86_64", but you have nvidia-cuda-cupti-cu12 12.4.127 which is incompatible.
torch 2.1.2 requires nvidia-cuda-runtime-cu12==12.1.105; platform_system == "Linux" and platform_machine == "x86_64", but you have nvidia-cuda-runtime-cu12 12.4.127 which is incompatible.
torch 2.1.2 requires nvidia-cufft-cu12==11.0.2.54; platform_system == "Linux" and platform_machine == "x86_64", but you have nvidia-cufft-cu12 11.2.1.3 which is incompatible.
torch 2.1.2 requires nvidia-cusolver-cu12==11.4.5.107; platform_system == "Linux" and platform_machine == "x86_64", but you have nvidia-cusolver-cu12 11.6.1.9 which is incompatible.
torch 2.1.2 requires nvidia-cusparse-cu12==12.1.0.106; platform_system == "Linux" and platform_machine == "x86_64", but you have nvidia-cusparse-cu12 12.3.1.170 which is incompatible.
torch 2.1.2 requires nvidia-nccl-cu12==2.18.1; platform_system == "Linux" and platform_machine == "x86_64", but you have nvidia-nccl-cu12 2.21.5 which is incompatible.



## Verify that Jax is working

1. Start a python interactive session `python`
2. `import jax`
3. `jax.numpy.array([1,]).devices()` should show that it's on cuda instead of cpu.

### Debugging jax

jaxlib.xla_extension.XlaRuntimeError: INTERNAL: XLA requires ptxas version 11.8 or higher

- Remember to export `export PATH=<path to your cuda environment>:$PATH`. For example, `export PATH=/home/hw575/.conda/envs/crossq/bin:$PATH`

Some error like this: "CUDA backend failed to initialize: Found CUDA version 12010, but JAX was built against version 12020, which is newer. The copy of CUDA that is installed must be at least as new as the version against which JAX was built. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)"

- Try running this:
```
pip install nvidia-cublas-cu12==12.4.2.65 nvidia-cuda-cupti-cu12==12.4.99 nvidia-cuda-nvrtc-cu12==12.4.99 nvidia-cuda-runtime-cu12==12.4.99 nvidia-cudnn-cu12==8.9.7.29 nvidia-cufft-cu12==11.2.0.44 nvidia-cusolver-cu12==11.6.0.99 nvidia-cusparse-cu12==12.3.0.142 nvidia-nccl-cu12==2.20.5
```

# How to train in HumanoidEnv

## Training
### Hand-engineered reward
To train with ground-truth geom xpos (only the arm's joint position), you have to specify
- the env's task name
- the env's reward type
```bash
python train.pyreward_model=hand_engineered env.reward_type="goal_only_euclidean_geom_xpos" env.task_name='right_arm_extend_wave_higher'
``` 

### Seq-matching reward
To train with a seq-matching reward, you have to specify
- the env's task name
- the env's reward type (which is just the standing up reward)
- the reward model
    - the gamma of the reward model
    - the reward model's cost function
    - the reward model's seq name
    - the reward model's reward vmin and vmax
        Only used to visualize the reward as the heat map (It allows us to visualize the reward across different rollouts with the same scale)
    - If we are doing post processing
        - "exp_reward" is taking the exponential of the cost from the sequence matching function
        - "stage_reward_based_on_last_state" is giving a bonus reward if the last state is the same as the last state of the sequence
            Warning: You must define a stage_bonus
```bash
python train.py env.reward_type="basic_r_geom_xpos" reward_model=soft_dtw reward_model.gamma=5 reward_model.cost_fn=euclidean_arms_only +reward_model.stage_bonus=0 reward_model.reward_vmin=0 reward_model.reward_vmax=1 '+reward_model.post_processing_method=["​​exp_reward", "stage_reward_based_on_last_state"]' env.task_name='right_arm_extend_wave_higher' reward_model.seq_name='key_frames' 
``` 

**[An example use case for using stage_bonus != 0]**
Because the seq matching fn outputs - cost, the reward is negative. When we give the bonus reward, the code does `reward_bonus += stage_bonus + reward[i-1]`. If reward[i-1] is negative, the bonus reward is negative, which is not great. Instead, we can set a stage_bonus to be positive, so that the bonus reward can be more positive.

```bash
python train.py env.reward_type="basic_r_geom_xpos" reward_model=soft_dtw reward_model.gamma=5 reward_model.cost_fn=euclidean_arms_only '+reward_model.post_processing_method=["stage_reward_based_on_last_state"]' +reward_mode.stage_bonus=2  reward_model.seq_name='key_frames' 'run_notes="debug-with-key-frames"'
```

## Inference
The rollouts/videos are saved in training logs.

Note. If you want to specify the `model_base_path`, because the model_base_path contains `=` which is how hydra uses to identify argument assignment, you need to wrap the model_base_path in quotes, i.e. `'model_base_path="<path to model folder>"'`
```bash
python inference.py 'model_base_path="train_logs/2024-08-14-120406_crossq_envr=both_arms_out_goal_only_euclidean_rm=dino_patch_wasserstein_s=9_nt=None/checkpoint"' model_checkpoint="model_2000_steps" env.reward_type="both_arms_out_goal_only_euclidean"
```


## Sometimes ctrl-c doesn't exit...
When you ctrl-c sometimes, the progress bar might keep appear when you type.
1. `nvidia-smi` to find the process that is running
2. `kill -9 <pid>` where pid is the process that you need to kill


# Metaworld
## Installation
1. If you have just cloned the repo, make sure to run the following command at the root directory ("CrossQ"):
```
git submodule update --init
```
2. `cd envs/Metaworld` and run `pip install -e .`

3. To make the gym version play well with the rest of the code, you need to reinstall gymnasium
```
pip install gymnasium==0.29.1
```

## Training in Metaworld
To use Metaworld, you have to either:
- change the defaults env (in train_config.yaml) to Metaworld
- or specify the env in the command line
```bash
python train.py env=Metaworld ...
```

### Selecting Tasks
There are 2 types of tasks:
- For making the goal observable, the task name should end with `goal-observable` (e.g., `button-press-v2-goal-observable`)
- For making the goal not observable, the task name should end with `goal-hidden` (e.g., `button-press-v2-goal-hidden`)

For example:
```bash
python train.py env=Metaworld env.task_name='button-press-v2-goal-observable' ...
```

### Reward from the environment
There are 3 options:
- 'dense': based on the reward function defined by Metaworld environment
- 'sparse': binary, based on the success of the task
- 'none': always 0

For example:
```bash
python train.py env=Metaworld env.env_reward_type='sparse' env.task_name='button-press-v2-goal-observable' ...
```

### Modifying the environment

- Main training script defines a make_env_fn, which returns an instantation of an environment class
- The environment class comes from the task name and the environments defined in metaworld.envs.mujoco.env_dict
```python
return Monitor(env_cls_to_use[cfg.env.task_name](render_mode="rgb_array", 
                                camera_name=cfg.env.camera_name,
                                episode_length=cfg.env.episode_length,
                                # Change the dense reward to sparse reward
                                env_reward_type=cfg.env.env_reward_type if "env_reward_type" in cfg.env else "dense",
                                temporal_encoding=cfg.env.temporal_encoding,))
```
- metaworld.envs.mujoco.env_dict._create_hidden_goal_envs, _create_observable_goal_envs define functions that create env classes, which are then instantiated through make_vec_env (in the main training script)
    - Any changes to env classes must also appear in initialize()
    - For example, to add an argument like “temporal_encoding”, it must be added to initialize(), and also added in the super() call (in both hidden and observable, if you plan on using both)
    
```python
def initialize(env, seed=None, render_mode=None, 
                camera_name="corner", 
                episode_length=200, 
                env_reward_type="none",
                temporal_encoding=False): # IMPORTANT: add new kwarg
    if seed is not None:
        st0 = np.random.get_state()
        np.random.seed(seed)

    super(type(env), env).__init__(temporal_encoding=temporal_encoding) # IMPORTANT: call superclass with kwargs to modify environment
```
    
- super() will call the sawyer environment class (e.g. SawyerButtonPressEnvV2) 
    - **IMPORTANT: for new envs to use new arguments, must add **kwargs to the environment __init__**

```python

class SawyerButtonPressEnvV2(SawyerXYZEnv):
    def __init__(
        self,
        render_mode: RenderMode | None = None,
        camera_name: str | None = None,
        camera_id: int | None = None,
        **kwargs # IMPORTANT: MUST ADD THIS (does not exist by default)
    ) -> None:
		   ...
	     super().__init__(
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
            camera_name=camera_name,
            camera_id=camera_id,
            **kwargs # IMPORTANT: must also pass them to the superclass
        )
```
- Sawyer environment class will instantiate superclass, which is SawyerXYZEnv
- SawyerXYZEnv handles metaworld observation space setup, env steps, resets, default (eg. grasping based) rewards, and specific environment classes (e.g. SawyerButtonPressEnvV2) extend it

## Sequence Matching Reward in Metaworld

### Setting the sequences
Similar to the Humanoid env, all the reference sequence are stored in `constants.py` in dictionary `METAWORLD_TASK_SEQ_DICT`

An example entry for a task is (TODO: the path is deprecated)
```bash
"button-press-v2":
    {
        "task_type": "goal_reaching",
        "sequences": {
            "rl_expert": "/share/portal/hw575/CrossQ/train_logs/2024-11-25-125324_sb3_sac_envt=button-press-v2-goal-hidden_rm=hand_engineered_nt=ep-len=200_sparse/eval/1000000_rollouts_states.npy"
        }
    }
```

### Training with (state-based) sequence matching reward
To train with a seq-matching reward, you have to specify the reward model. (And we can also set the environment's reward type to none)

```bash
python train.py env.env_reward_type='none' env.task_name=button-press-v2-goal-hidden reward_model=testing_dist_metric
```

### Creating your own sequence matching reward function
These are the steps to create your own sequence matching reward function:
1. Create a yaml file in `configs/reward_models`. You are required to have the following field
```yaml
name: testing_dist_metric # str
cost_fn: euclidean # str (it has to match the key in the `COST_FN_DICT` in `seq_reward/cost_fns.py`)
seq_name: rl_expert # str (it has to match the key under "sequences" in `METAWORLD_TASK_SEQ_DICT` in `constants.py`)
```
2. Define the reward model function in `seq_reward/` folder. The function must follow these requirements:
```python
"""
Parameters:
    obs: np.ndarray
        The observed sequence of joint states
        size: (train_freq, 22)
            For OT-based reward, train_freq == episode_length
            22 is the observation size that we want to calculate
    ref: np.ndarray
        The reference sequence of joint states
        size: (ref_seq_len, 22)
            22 is the observation size that we want to calculate
    additional parameters...

Returns:
    reward: np.ndarray
        The reward for each frame in the observed sequence
        size: (train_freq, )
    info: dict
        Required to have the following (for downstream visualization)
            - cost_matrix: np.ndarray (train_freq, ref_seq_len)
            - assignment_matrix: np.ndarray (train_freq, ref_seq_len)
"""
```
3. In `seq_reward/seq_utils.py` define how the sequence reward function will be loaded in `get_matching_fn()` 
4. In `utils.py` add the name of the reward model to the `use_sequence_matching_fn_for_reward()` function
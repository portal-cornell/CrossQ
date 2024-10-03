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

# How to train

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
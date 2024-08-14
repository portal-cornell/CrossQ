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
Using all default hydra parameters
```bash
python train.py 
```

For training with hand-engineered reward, you can edit the yaml or directly pass the arguments in the command line.
```bash
python train.py env.reward_type='both_arms_out_goal_only_euclidean'
``` 

For training VLM, you can edit the yaml or directly pass the arguments in the command line.
```bash
python train.py env.reward_type='both_arms_out_goal_only_euclidean' reward_model=patch_wasserstein
``` 


## Sometimes ctrl-c doesn't exit...
When you ctrl-c sometimes, the progress bar might keep appear when you type.
1. `nvidia-smi` to find the process that is running
2. `kill -9 <pid>` where pid is the process that you need to kill
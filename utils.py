import datetime
import secrets
import os

import itertools
from typing import Any, Callable, Dict, List, Optional, Type, Union

from constants import WANDB_DIR

def get_run_hash() -> str:
    return f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_', f"{secrets.token_hex(4)}"

def vlm_for_reward(args) -> bool:
    return args.reward_model_name != ""

def set_os_vars() -> None:
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    # Get egl (mujoco) rendering to work on cluster
    os.environ["NVIDIA_VISIBLE_DEVICES"] = "all"
    os.environ["NVIDIA_DRIVER_CAPABILITIES"] = "compute,graphics,utility,video"
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["EGL_PLATFORM"] = "device"
    # Get wandb file (e.g. rendered) gif more accessible
    os.environ["WANDB_DIR"] = WANDB_DIR
    os.environ["LOGURU_LEVEL"] = "INFO"

def validate_args(args):
    if vlm_for_reward(args):
        assert args.reward_batch_size % args.n_workers == 0, f"({args.reward_batch_size=}) corresponds to the total size of the batch do be distributed among workers and therefore must be divisible by ({args.n_workers=})"

        assert (args.n_envs * args.episode_length) % args.reward_batch_size == 0, f"({args.n_envs=}) * ({args.episode_length=}) must be divisible by ({args.reward_batch_size=}) so that all batches are of the same size."

def get_model_args_dict(args):
    import jax
    
    """
    Effect: Modify args
    """
    args.algo = str.lower(args.algo)
    args.bn = bool(args.bn)
    args.crossq_style = bool(args.crossq_style)
    args.tau = float(args.tau) if not args.crossq_style else 1.0
    args.bn_momentum = float(args.bn_momentum) if args.bn else 0.0
    dropout_rate, layer_norm = None, False
    policy_q_reduce_fn = jax.numpy.min
    net_arch = {'pi': [256, 256], 'qf': [args.n_neurons, args.n_neurons]}

    args.total_timesteps = int(args.total_timesteps)

    td3_mode = False

    if args.algo == 'droq':
        dropout_rate = 0.01
        layer_norm = True
        policy_q_reduce_fn = jax.numpy.mean
        args.n_critics = 2
        # args.adam_b1 = 0.9  # adam default
        args.adam_b2 = 0.999  # adam default
        args.policy_delay = 20
        args.utd = 20
        group = f'DroQ_{args.env}_bn({args.bn})_ln{(args.ln)}_xqstyle({args.crossq_style}/{args.tau})_utd({args.utd}/{args.policy_delay})_Adam({args.adam_b1})_Q({net_arch["qf"][0]})'

    elif args.algo == 'redq':
        policy_q_reduce_fn = jax.numpy.mean
        args.n_critics = 10
        # args.adam_b1 = 0.9  # adam default
        args.adam_b2 = 0.999  # adam default
        args.policy_delay = 20
        args.utd = 20
        group = f'REDQ_{args.env}_bn({args.bn})_ln{(args.ln)}_xqstyle({args.crossq_style}/{args.tau})_utd({args.utd}/{args.policy_delay})_Adam({args.adam_b1})_Q({net_arch["qf"][0]})'

    elif args.algo == 'td3':
        # With the right hyperparameters, this here can run all the above algorithms
        # and ablations.
        td3_mode = True
        layer_norm = args.ln
        if args.dropout: 
            dropout_rate = 0.01
        group = f'TD3_{args.env}_bn({args.bn}/{args.bn_momentum}/{args.bn_mode})_ln{(args.ln)}_xq({args.crossq_style}/{args.tau})_utd({args.utd}/{args.policy_delay})_A{args.adam_b1}_Q({net_arch["qf"][0]})_l{args.lr}'

    elif args.algo == 'sac':
        # With the right hyperparameters, this here can run all the above algorithms
        # and ablations.
        layer_norm = args.ln
        if args.dropout: 
            dropout_rate = 0.01
        group = f'SAC_{args.env}_bn({args.bn}/{args.bn_momentum}/{args.bn_mode})_ln{(args.ln)}_xq({args.crossq_style}/{args.tau})_utd({args.utd}/{args.policy_delay})_A{args.adam_b1}_Q({net_arch["qf"][0]})_l{args.lr}'

    elif args.algo == 'crossq':
        args.adam_b1 = 0.5
        args.policy_delay = 3
        args.n_critics = 2
        args.utd = 1                    # nice
        net_arch["qf"] = [2048, 2048]   # wider critics
        args.bn = True                  # use batch norm
        args.crossq_style = True        # with a joint forward pass
        args.tau = 1.0                  # without target networks
        group = f'CrossQ_{args.env}'

    else:
        raise NotImplemented

    args_dict = vars(args)
    args_dict.update({
        "dropout_rate": dropout_rate,
        "layer_norm": layer_norm,
        "net_arch": net_arch,
        "td3_mode": td3_mode,
        "policy_q_reduce_fn": policy_q_reduce_fn
    })

    return args, args_dict
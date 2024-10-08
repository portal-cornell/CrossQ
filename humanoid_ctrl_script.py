"""
Use Cases:
1. Generate data for a single pose
    python humanoid_ctrl_script.py -p=both-arms-out

    if --debug is set, a video will be saved in debugging/humanoid_env/testing.mp4
2. Generate data for a sequence of poses
    python humanoid_ctrl_script.py --gen_traj --seq_name=both-arms-out_to_left-arm-out --steps=60

    if --debug is set, a video will be saved in debugging/humanoid_env/testing.mp4
3. Generate data for a sequence of poses and take the last n frames
    python humanoid_ctrl_script.py --gen_traj --seq_name=both-arms-out_to_left-arm-out --steps=20 --take_last_n_frames=10

    if --debug is set, a video will be saved in debugging/humanoid_env/testing.mp4
"""

import gymnasium
from loguru import logger
import os
import imageio
import numpy as np
import copy
from PIL import Image
import argparse

import envs
from create_demo.pose_config import pose_config_dict
from inference import plot_info_on_frame

# Set up
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# Get egl (mujoco) rendering to work on cluster
os.environ["NVIDIA_VISIBLE_DEVICES"] = "all"
os.environ["NVIDIA_DRIVER_CAPABILITIES"] = "compute,graphics,utility,video"
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["EGL_PLATFORM"] = "device"

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--pose_name",   type=str,   required=False, help="Name of the demo to generate data for (correspond to create_demo/pose_config.py)")
parser.add_argument("--debug", default=False, action="store_true", help="Store a video and add more prints to help visualize the data")
# For generating an entire trajectory through interpolation
parser.add_argument("--gen_traj", default=False, action="store_true", help="Generate a trajectory through interpolation for the pose")
parser.add_argument("--seq_name", default="", type=str, help="Name of the sequence of poses to generate")
parser.add_argument("--steps", default=60, type=int, help="Number of steps to interpolate between poses")
parser.add_argument("--take_last_n_frames", default=-1, type=int, help="After interpolating {# of key poses} * {steps} frames, we use the last n frames to save as ref seq")


args = parser.parse_args()

if not args.gen_traj:
    assert args.pose_name in pose_config_dict, f"-pose_name={args.pose_name} does not match any of the key in pose_config_dict in create_demo/pose_config.py"

if args.debug:
    # Initialize video writer for debugging 
    video_writer = imageio.get_writer("debugging/humanoid_env/testing.mp4", fps=30)

# Load the humanoid environment
make_env_kwargs = dict(
    episode_length = 120,
    reward_type = "original"
)

env = gymnasium.make(
            'HumanoidSpawnedUpCustom',
            render_mode="rgb_array",
            **make_env_kwargs,
        )

# Reset the environment
env.reset()

init_qpos = env.unwrapped.init_qpos
init_geom_pos = copy.deepcopy(env.unwrapped.data.geom_xpos)

if not args.gen_traj:
    print(f"env.init_qpos=({init_qpos.shape}), \n{init_qpos}")
    print(f"env.init_geom_pos=({init_geom_pos.shape}), \n{init_geom_pos}")

    if args.debug:
        video_writer.append_data(np.uint8(env.render()))

        init_qvel = env.unwrapped.init_qvel
        print(f"env.init_qvel=({init_qvel.shape}), \n{init_qvel}")

        obs = env.unwrapped.get_obs()
        print(f"before: {obs.shape}\n{obs[:22].shape}\n{obs[:22]}")

    # Set the humanoid to the pose specified in pose_config_dict
    joint_config = pose_config_dict[args.pose_name]

    new_qpos = copy.deepcopy(init_qpos)
    for idx in joint_config.keys():
        new_qpos[int(idx)] = joint_config[int(idx)] # z-coordinate of torso

    # Set the humanoid joint state to the specified ones, velocity stays at 0
    env.unwrapped.set_state(qpos=new_qpos, qvel=np.zeros((23,)))


    obs = env.unwrapped.get_obs()
    with open(f"create_demo/demos/{args.pose_name}_joint-state.npy", "wb") as fout:
        print(f"Obs after setting the pose: {obs.shape}\n{obs[:22].shape}\n{obs[:22]}")
        # Based on https://www.gymlibrary.dev/environments/mujoco/humanoid/ and
        #   https://docs.google.com/spreadsheets/d/17xxmlh8oLAh7vLlRGtMz4b79ueb0KwccQIvbBa0m3kk/edit?usp=sharing
        np.save(fout, obs[:22])

    with open(f"create_demo/demos/{args.pose_name}_geom-xpos.npy", "wb") as fout:
        print(f"gemo_xpos after setting the pose: {env.unwrapped.data.geom_xpos.shape}\n{env.unwrapped.data.geom_xpos}")
        #   https://docs.google.com/spreadsheets/d/17xxmlh8oLAh7vLlRGtMz4b79ueb0KwccQIvbBa0m3kk/edit?usp=sharing
        np.save(fout, copy.deepcopy(env.unwrapped.data.geom_xpos))

    # Render the environment to visualize the pose
    frame = env.render()
    image = Image.fromarray(frame)
    image.save(f"create_demo/demos/{args.pose_name}.png")

    if args.debug:
        video_writer.append_data(np.uint8(frame))

        # Check for pose's stability
        # Action space size from https://www.gymlibrary.dev/environments/mujoco/humanoid/
        for i in range(20):
            # Test how stable the pose is (0 force/action)
            env.step(np.zeros((17,)))
            video_writer.append_data(np.uint8(env.render()))

        video_writer.close()
else:
    seq_name_to_subgoal_list = {
        "right-arm-extend-wave-higher": ["right-arm-extend-wave-higher"],
        "left-arm-extend-wave-higher": ["left-arm-extend-wave-higher"],
        "left-arm-out": ["left-arm-out"],
        "left-arm-out_to_both-arms-out": ["left-arm-out", "both-arms-out"],
        "left-arm-out_to_right-arm-out": ["left-arm-out", "right-arm-out"],
        "right-arm-out_to_left-arm-out": ["right-arm-out", "left-arm-out"],
        "right-arm-out_to_both-arms-out": ["right-arm-out", "both-arms-out"],
        "both-arms-out_to_left-arm-out": ["both-arms-out", "left-arm-out"],
        "both-arms-out_to_right-arm-out": ["both-arms-out", "right-arm-out"],
        "left-arm-out_to_left-arm-extend-wave-higher": ["left-arm-out", "left-arm-extend-wave-higher"],
        "left-arm-lower_to_left-arm-extend-wave-higher": ["left-arm-lower", "left-arm-extend-wave-higher"],
    }

    assert args.seq_name in seq_name_to_subgoal_list, f"seq_name={args.seq_name} does not match any of the key in seq_name_to_subgoal_list"

    seq_name = args.seq_name
    subgoal_list = seq_name_to_subgoal_list[seq_name]

    num_of_steps_per_subgoal = args.steps

    print(f"Generating a trajectory for {seq_name} with {num_of_steps_per_subgoal} steps per subgoal\nsubgoal_list: {subgoal_list}")

    # Data to save
    frames = []
    qposes_to_save = []
    geom_xposes_to_save = []

    # A hack to make the initial qpos already spawn the humanoid landed on the floor
    init_qpos[2] = 1.3

    # Set the humanoid joint state to the specified ones, velocity stays at 0
    env.unwrapped.set_state(qpos=init_qpos, qvel=np.zeros((23,)))

    frames.append(Image.fromarray(env.render()))

    if args.debug:
        video_writer.append_data(np.uint8(env.render()))

    def from_init_qpos_to_new_qpos(init_qpos, new_qpos, step):
        """
        Interpolate between init_qpos and new_qpos.
        
        Returns:
            (step, qpos.shape) np.array
                Interpolated qpos to go from init_qpos to new_qpos in step steps
        """
        return np.linspace(init_qpos, new_qpos, step)
    
    for subgoal in subgoal_list:
        # Set the humanoid to the pose specified in pose_config_dict
        joint_config = pose_config_dict[subgoal]

        new_qpos = copy.deepcopy(init_qpos)
        for idx in joint_config.keys():
            new_qpos[int(idx)] = joint_config[int(idx)] # z-coordinate of torso

        interpolated_qposes= from_init_qpos_to_new_qpos(init_qpos, new_qpos, num_of_steps_per_subgoal)

        for qpos in interpolated_qposes:
            # Set the humanoid joint state to the specified ones, velocity stays at 0
            env.unwrapped.set_state(qpos=qpos, qvel=np.zeros((23,)))

            frames.append(Image.fromarray(env.render()))
            obs = env.unwrapped.get_obs()
            qposes_to_save.append(copy.deepcopy(obs[:22]))
            geom_xpos = copy.deepcopy(env.unwrapped.data.geom_xpos)
            # Normalize the geom_xpos to the torso
            geom_xpos -= geom_xpos[1]
            geom_xposes_to_save.append(copy.deepcopy(geom_xpos))

            if args.debug:
                video_writer.append_data(np.uint8(env.render()))

        init_qpos = new_qpos

    if args.take_last_n_frames != -1:
        frames = [frames[0]] + frames[-args.take_last_n_frames:]
        qposes_to_save = [qposes_to_save[0]] + qposes_to_save[-args.take_last_n_frames:]
        geom_xposes_to_save = [geom_xposes_to_save[0]] + geom_xposes_to_save[-args.take_last_n_frames:]
        save_keyword = "last-"
    else:
        save_keyword = ""
        
    ref_seq_len = len(frames) - 1 # Subtract the initial frame

    # Save the raw_screens locally
    imageio.mimsave(os.path.join("create_demo/seq_demos", f"{seq_name}_{save_keyword}{ref_seq_len}-frames.gif"), frames, duration=1/30, loop=0)
    with open(os.path.join("create_demo/seq_demos", f"{seq_name}_{save_keyword}{ref_seq_len}-frames_joint-states.npy"), "wb") as fout:
        print(f"qposes_to_save: {np.array(qposes_to_save).shape}")
        np.save(fout, np.array(qposes_to_save))

    with open(os.path.join("create_demo/seq_demos", f"{seq_name}_{save_keyword}{ref_seq_len}-frames_geom-xpos.npy"), "wb") as fout:
        print(f"geom_xposes_to_save: {np.array(geom_xposes_to_save).shape}")
        np.save(fout, np.array(geom_xposes_to_save))

    if args.debug:
        video_writer.close()

# Close the environment
env.close()

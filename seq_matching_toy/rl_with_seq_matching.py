from seq_matching_toy.toy_envs.grid_nav import *

if __name__ == "__main__":
    env = GridNavigationEnv(map_config={"name": "3x3"}, ref_seq=[], render_mode="rgb_array")
    env.reset()
    env.render()
    
    path = [RIGHT, RIGHT, DOWN, DOWN, LEFT, LEFT, UP, STAY]

    frames = []
    frames.append(env.render())

    for action in path:
        env.step(action)
        frames.append(env.render())

    imageio.mimsave("testing.gif", frames, duration=1/20, loop=0)

    writer = imageio.get_writer('testing.mp4', fps=20)

    for im in frames:
        writer.append_data(im)
    
    writer.close()
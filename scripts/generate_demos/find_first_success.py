import numpy as np

def find_and_write_consecutive_ones(paths, output_file):
    results = {}
    for path in paths:

        success_path = path.replace("_states", "_success")
        data = np.load(success_path)
        # Extract the environment name (e.g., "lever-pull")
        env_name = path.split('/')[-2].split('-v2')[0]
        
        if len(data) < 5:
            results[env_name] = None
            continue
        for i in range(len(data) - 4):
            if np.all(data[i:i+5] == 1):
                results[env_name] = i
                break
        else:
            results[env_name] = None
    
    # Write the results to the output file
    with open(output_file, 'w') as f:
        for env_name, idx in results.items():
            f.write(f"{env_name}: {{idx: {idx}}}\n")


if __name__ == "__main__":
    # Example usage
    paths = [
        "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/button-press-v2/button-press-v2_corner_0_states.npy",
        "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/door-close-v2/door-close-v2_corner_0_states.npy",
        "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/door-open-v2/door-open-v2_corner3_0_states.npy",
        "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/window-open-v2/window-open-v2_corner3_0_states.npy",
        "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/lever-pull-v2/lever-pull-v2_corner4_0_states.npy",
        "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/hand-insert-v2/hand-insert-v2_corner_0_states.npy",
        "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/push-v2/push-v2_corner3_0_states.npy",
        "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/basketball-v2/basketball-v2_corner_0_states.npy",
        "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/stick-push-v2/stick-push-v2_corner_0_states.npy",
        "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/door-lock-v2/door-lock-v2_corner_0_states.npy",
        "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/bin-picking-v2/bin-picking-v2_corner_0_states.npy",
        "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/box-close-v2/box-close-v2_corner3_0_states.npy",
        "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/pick-place-v2/pick-place-v2_corner3_0_states.npy",
        "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/assembly-v2/assembly-v2_corner_0_states.npy",
        "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/disassemble-v2/disassemble-v2_corner_0_states.npy",
        "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/hammer-v2/hammer-v2_corner3_0_states.npy",
        "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/peg-insert-side-v2/peg-insert-side-v2_corner3_0_states.npy",
        "/share/portal/hw575/CrossQ/create_demo/metaworld_demos/door-unlock-v2/door-unlock-v2_corner_0_states.npy"]

    output_file = "output.txt"
    find_and_write_consecutive_ones(paths, output_file)
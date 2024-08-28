WANDB_DIR = "./"

# Used abosolute path because eval is run in a subdirectory
# TODO: There's probably a better way to do this
SEQ_DICT = {
        "both_arms_out": ["/share/portal/hw575/CrossQ/create_demo/demos/both-arms-out_joint-state.npy"],
        "both_arms_out_with_intermediate": ["/share/portal/hw575/CrossQ/create_demo/demos/left-arm-out_joint-state.npy", "/share/portal/hw575/CrossQ/create_demo/demos/both-arms-out_joint-state.npy"],
        "arms_up_then_down": ["/share/portal/hw575/CrossQ/create_demo/demos/left-arm-out_joint-state.npy", "/share/portal/hw575/CrossQ/create_demo/demos/both-arms-out_joint-state.npy", "/share/portal/hw575/CrossQ/create_demo/demos/right-arm-out_joint-state.npy"],
    }
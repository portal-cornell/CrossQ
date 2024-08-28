import numpy as np

"""++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Helper functions

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"""

SITUP_HEIGHT = 0.5  # From looking at the environment
SITUP_TOLERANCE = 0.1
# 1.4 is from humanoid.xml and where they spawn the humanoid (should be slightly higher than the default height)
# since 1.3 is mentioned in the paper (https://ieeexplore-ieee-org.proxy.library.cornell.edu/stamp/stamp.jsp?tp=&arnumber=6386025)
STANDUP_HEIGHT = 1.3 

GEOM_XPOS_NAMING = {0: "floor", 1:"torso", 2:"head", 3:"uwaist", 4:"lwaist", 5:"bottom",
                6:"R_thigh", 7:"R_shin", 8:"R_foot",
                9:"L_thigh", 10:"L_shin", 11:"L_foot",
                12:"R_uarm", 13:"R_larm", 14:"R_hand",
                15:"left_uarm", 16:"L_arm", 17:"L_hand"}


def basic_remain_standing_rewards(data, 
                                    upward_reward_w=1, 
                                    ctrl_cost_w=1, 
                                    **kwargs):
    """Basic reward function for the humanoid to remain standing

    Parameters:
        data (mujoco data): the data from the mujoco environment
        upward_reward_w (float): the weight for the upward reward
        ctrl_cost_w (float): the weight for the control cost
        **kwargs: additional arguments

    Returns:
        float: the reward for the humanoid to remain standing (upward reward - control cost)
    """
    ctrl_cost = kwargs.get("ctrl_cost", None)
    ctrl_cost_w = 1

    upward_reward = np.exp(-(data.qpos.flat[2] - 1.3)**2)
    upward_reward_w = 1

    terms_to_plot = dict(
        tor=str([f"{data.qpos.flat[:3][i]:.2f}" for i in range(3)]),
        com=str([f"{data.xipos[1][i]:.2f}" for i in range(3)]),
        uph_r= f"{upward_reward:.2f}",
        ctrl_c= f"{ctrl_cost:.2f}",
    )

    return upward_reward_w * upward_reward - ctrl_cost_w * ctrl_cost, terms_to_plot


def smooth_abs_norm(x, alpha=1.0):
    # on page 3 of the paper (https://ieeexplore-ieee-org.proxy.library.cornell.edu/stamp/stamp.jsp?tp=&arnumber=6386025), mentioned starting alpha=1 then slowly decreasing
    return np.sqrt(x**2 + alpha**2) - alpha

def control_limiting_cost(x, alpha=1.0):
    return alpha ** 2 * (np.cosh(x / alpha) - 1)

def upward_reward(data, timestep):
    """A reward for moving upward (in an attempt to stand up)
    """
    com_z = data.qpos[2]
    return (com_z - 0) / timestep


def control_cost(data):
    """Penalizing the humanoid if it has too large of a control force.
    """
    return 0.1 * np.square(data.ctrl).sum()


def impact_cost(data):
    """Penalizing the humanoid if the external contact force is too large.
    """
    quad_impact_cost = 0.5e-6 * np.square(data.cfrc_ext).sum()
    return min(quad_impact_cost, 10)


def feet_are_above_com_cost(data):
    """Penalizing the cost of putting feet above com
    """
    com_z = data.qpos[2]
    left_foot_z = data.geom_xpos[11][2]
    right_foot_z = data.geom_xpos[8][2]

    # When the humanoid is resting, upward reward is still about 40
    #   It will be suboptimal to just keep my feet up in the air (max=0.75) while my com (0.1) is on the ground
    left_foot_cost = 100 * (left_foot_z > com_z) * (left_foot_z - com_z)
    right_foot_cost = 100 * (right_foot_z > com_z) * (right_foot_z - com_z)
    # We want the maximum value to be bad, but not too bad (TODO: tune these values)
    return min(left_foot_cost + right_foot_cost, 100)


def is_sitting_up(data):
    com_z = data.qpos[2]
    return np.abs(com_z - SITUP_HEIGHT) < SITUP_TOLERANCE


def dist_btw_com_and_feet_cost(data, timestep):
    """Penalising the cost between the distance between the center of mass and feet
    """
    com_xy = data.qpos[:2]
    left_foot_xy = data.geom_xpos[11][:2]
    right_foot_xy = data.geom_xpos[8][:2]

    feet_midpt = np.array([(left_foot_xy[0]-right_foot_xy[0])/2.0, (left_foot_xy[1]-right_foot_xy[1])/2.0])

    left_foot_dist = np.sqrt(np.sum((com_xy - left_foot_xy)**2))
    right_foot_dist = np.sqrt(np.sum((com_xy - right_foot_xy)**2))
    feet_midpt_dist = np.sqrt(np.sum((com_xy - feet_midpt)**2))

    terms_to_plot = dict(
        # feet_midpt = str([f"{feet_midpt[i]:.2f}" for i in range(len(feet_midpt))]),
        feet_midpt_dist = f"{feet_midpt_dist:.2f}",
        left_foot_dist = f"{left_foot_dist:.2f}",
        right_foot_dist = f"{right_foot_dist:.2f}",
    )

    # TODO: Remove debugging terms to plot
    # return dist / timestep, terms_to_plot
    scale = 15
    return min(scale * (left_foot_dist + right_foot_dist + feet_midpt_dist), scale*3), terms_to_plot

def dist_btw_com_and_feet_cost_v1(data, timestep):
    """Penalising the cost between the distance between the center of mass and feet
    """
    com_xy = data.qpos[:2]
    left_foot_xy = data.geom_xpos[11][:2]
    right_foot_xy = data.geom_xpos[8][:2]

    feet_midpt = np.array([(left_foot_xy[0]-right_foot_xy[0])/2.0, (left_foot_xy[1]-right_foot_xy[1])/2.0])

    left_foot_dist = np.sqrt(np.sum((com_xy - left_foot_xy)**2))
    right_foot_dist = np.sqrt(np.sum((com_xy - right_foot_xy)**2))
    # feet_midpt_dist = np.sqrt(np.sum((com_xy - feet_midpt)**2))

    terms_to_plot = dict(
        # feet_midpt = str([f"{feet_midpt[i]:.2f}" for i in range(len(feet_midpt))]),
        # feet_midpt_dist = f"{feet_midpt_dist:.2f}",
        left_foot_dist = f"{left_foot_dist:.2f}",
        right_foot_dist = f"{right_foot_dist:.2f}",
    )

    # TODO: Remove debugging terms to plot
    # return dist / timestep, terms_to_plot
    scale = 20
    return min(scale * (left_foot_dist + right_foot_dist), scale*2), terms_to_plot

def hori_dist_btw_com_and_mean_feet(data):
    # based on https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjdata.h#L252 and the humanoid.xml file
    com_xy = data.xipos[1][:2]

    left_foot_xy = data.geom_xpos[11][:2]
    right_foot_xy = data.geom_xpos[8][:2]

    feet_midpt = np.array([(left_foot_xy[0]-right_foot_xy[0])/2.0, (left_foot_xy[1]-right_foot_xy[1])/2.0])

    terms_to_plot = dict(
        feet_midpt = str([f"{feet_midpt[i]:.2f}" for i in range(len(feet_midpt))]),
    )

    dist_cost = smooth_abs_norm(com_xy[0] - feet_midpt[0]) + smooth_abs_norm(com_xy[1] - feet_midpt[1])
    # TODO: Remove debugging terms to plot
    # return dist / timestep, terms_to_plot
    return dist_cost, terms_to_plot


def hori_dist_btw_com_and_mean_feet_no_smooth_abs(data):
    # based on https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjdata.h#L252 and the humanoid.xml file
    com_xy = data.xipos[1][:2]

    left_foot_xy = data.geom_xpos[11][:2]
    right_foot_xy = data.geom_xpos[8][:2]

    feet_midpt = np.array([(left_foot_xy[0]-right_foot_xy[0])/2.0, (left_foot_xy[1]-right_foot_xy[1])/2.0])

    terms_to_plot = dict(
        feet_midpt = str([f"{feet_midpt[i]:.2f}" for i in range(len(feet_midpt))]),
    )

    dist_cost = np.linalg.norm(com_xy - feet_midpt)
    # TODO: Remove debugging terms to plot
    # return dist / timestep, terms_to_plot
    return dist_cost, terms_to_plot

def hori_dist_btw_com_and_feet_no_smooth_abs(data):
    # based on https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjdata.h#L252 and the humanoid.xml file
    com_xy = data.xipos[1][:2]

    left_foot_xy = data.geom_xpos[11][:2]
    right_foot_xy = data.geom_xpos[8][:2]

    terms_to_plot = dict(
        right_foot_xy = str([f"{right_foot_xy[i]:.2f}" for i in range(len(right_foot_xy))]),
        left_foot_xy = str([f"{left_foot_xy[i]:.2f}" for i in range(len(left_foot_xy))]),
    )

    left_foot_dist_cost = np.linalg.norm(com_xy - left_foot_xy)
    right_foot_dist_cost = np.linalg.norm(com_xy - right_foot_xy)
    # TODO: Remove debugging terms to plot
    # return dist / timestep, terms_to_plot
    return left_foot_dist_cost, right_foot_dist_cost, terms_to_plot

def hori_dist_btw_torso_and_feet_no_smooth_abs(data):
    # based on https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjdata.h#L252 and the humanoid.xml file
    torso_xy = data.qpos[:2]

    left_foot_xy = data.geom_xpos[11][:2]
    right_foot_xy = data.geom_xpos[8][:2]

    terms_to_plot = dict(
        right_foot_xy = str([f"{right_foot_xy[i]:.2f}" for i in range(len(right_foot_xy))]),
        left_foot_xy = str([f"{left_foot_xy[i]:.2f}" for i in range(len(left_foot_xy))]),
    )

    left_foot_dist_cost = np.linalg.norm(torso_xy - left_foot_xy)
    right_foot_dist_cost = np.linalg.norm(torso_xy - right_foot_xy)
    # TODO: Remove debugging terms to plot
    # return dist / timestep, terms_to_plot
    return left_foot_dist_cost, right_foot_dist_cost, terms_to_plot

def hori_dist_btw_torso_and_com_circular_range_and_feet_no_smooth_abs(data):
    # based on https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjdata.h#L252 and the humanoid.xml file
    com_xy = data.xipos[1][:2]
    torso_xy = data.qpos[:2]

    radius = 0.3

    left_foot_xy = data.geom_xpos[11][:2]
    right_foot_xy = data.geom_xpos[8][:2]

    terms_to_plot = dict(
        right_foot_xy = str([f"{right_foot_xy[i]:.2f}" for i in range(len(right_foot_xy))]),
        left_foot_xy = str([f"{left_foot_xy[i]:.2f}" for i in range(len(left_foot_xy))]),
    )

    left_foot_torso_dist_cost = np.linalg.norm(torso_xy - left_foot_xy)
    right_foot_torso_dist_cost = np.linalg.norm(torso_xy - right_foot_xy)

    left_foot_torso_dist_cost = 0 if left_foot_torso_dist_cost < radius else left_foot_torso_dist_cost
    right_foot_torso_dist_cost = 0 if right_foot_torso_dist_cost < radius else right_foot_torso_dist_cost

    left_foot_com_dist_cost = np.linalg.norm(com_xy - left_foot_xy)
    right_foot_com_dist_cost = np.linalg.norm(com_xy - right_foot_xy)

    left_foot_com_dist_cost = 0 if left_foot_com_dist_cost < radius else left_foot_com_dist_cost
    right_foot_com_dist_cost = 0 if right_foot_com_dist_cost < radius else right_foot_com_dist_cost
    # TODO: Remove debugging terms to plot
    # return dist / timestep, terms_to_plot
    return left_foot_torso_dist_cost, right_foot_torso_dist_cost, left_foot_com_dist_cost, right_foot_com_dist_cost, terms_to_plot

def vert_dist_btw_bottom_and_feet(data):
    # based on https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjdata.h#L252 and the humanoid.xml file
    bottom_z = data.geom_xpos[5][2]

    left_foot_z = data.geom_xpos[11][2]
    right_foot_z = data.geom_xpos[8][2]

    terms_to_plot = dict(
        bottom_z = f"{bottom_z:.2f}",
    )

    # TODO: Remove debugging terms to plot
    # return dist / timestep, terms_to_plot
    return (bottom_z - left_foot_z), (bottom_z - right_foot_z), terms_to_plot

def hori_dist_btw_com_and_torso(data):
    com_xy = data.xipos[1][:2]
    torso_xy = data.qpos[:2]

    return smooth_abs_norm(com_xy[0] - torso_xy[0]) + smooth_abs_norm(com_xy[1] - torso_xy[1])

def hori_dist_btw_com_and_torso_no_smooth_abs(data):
    com_xy = data.xipos[1][:2]
    torso_xy = data.qpos[:2]

    return np.linalg.norm(com_xy - torso_xy)

def vert_dist_btw_torso_and_standing_height(data):
    torso_z = data.qpos[2]

    return smooth_abs_norm(torso_z - STANDUP_HEIGHT)

def hori_com_vel_cost(data):
    torso_xy_vel = data.qvel[:2]
    
    return control_limiting_cost(torso_xy_vel[0]) + control_limiting_cost(torso_xy_vel[1])

def dist_cost_with_tol(dist_with_tolerance_list):
    total_cost_list = []

    for i in range(len(dist_with_tolerance_list)):
        if len(dist_with_tolerance_list[i]) == 2:
            dist, tol = dist_with_tolerance_list[i]
            cost = 0 if dist < tol else dist
        elif len(dist_with_tolerance_list[i]) == 3:
            dist, tol, weight = dist_with_tolerance_list[i]
            cost = 0 if dist < tol else dist * weight
        else:
            assert len(dist_with_tolerance_list[i]) == 2 or len(dist_with_tolerance_list[i]) == 3, f"{dist_with_tolerance_list[i]} does not have the right format"

        total_cost_list.append(cost)

    return total_cost_list
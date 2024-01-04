# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class STOCH3ForceControlCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 17
        num_privileged_obs = None
        # if not None a privileged_obs_buf will be returned by step()
        # (critic obs for asymmetric training). None is returned otherwise
        num_actions = 6
        num_joints = 12
        env_spacing = 3.  # not used with height-fields/tri-meshes
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 20  # episode length in seconds

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.46]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quaternion]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = {'fl_abd_joint': 0.0, 'bl_abd_joint': 0.0,
                                'fr_abd_joint': 0.0, 'br_abd_joint': 0.0,
                                'fl_hip_joint': 0.8545, 'bl_hip_joint': 0.8545,
                                'fr_hip_joint': 0.8545, 'br_hip_joint': 0.8545,
                                'fl_knee_joint': -1.5563, 'bl_knee_joint': -1.5563,
                                'fr_knee_joint': -1.5563, 'br_knee_joint': -1.5563}

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.
        # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading
        # (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 3
        # time before command are changed[s]
        resampling_time = 10.
        # if true: compute ang vel command from heading error
        heading_command = False
        class ranges:
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-0.8, 0.8]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'  # 'T', 'V'
        controller = 'isaac'  # 'custom
        if controller == 'isaac':
            stiffness = {'joint': 400, 'cartesian': 100}  # [N*m/rad]
            damping = {'joint': 10, 'cartesian': 2}  # [N*m*s/rad]
        if controller == 'custom':
            stiffness = {'joint': 200, 'cartesian': 100}  # [N*m/rad]
            damping = {'joint': 2, 'cartesian': 1}     # [N*m*s/rad]
        if controller == 'custom':
            stiffness = {'joint': 200, 'cartesian': 100}  # [N*m/rad]
            damping = {'joint': 2, 'cartesian': 1.0}     # [N*m*s/rad]
        action_scale = 1.0
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 1

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/stoch3/urdf/stoch3.urdf'
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/stoch3/urdf/stoch3_dynamicWeight.urdf'
        name = "stoch3"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "shank", "abd"]
        terminate_after_contacts_on = ["base"]  # ["base","abd"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        collapse_fixed_joints = False
        flip_visual_attachments = False
        # Some .obj meshes must be flipped from y-up to z-up
        default_dof_drive_mode = 0
        # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        fix_base_link = False
        # fix the base of the robot

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = False
        friction_range = [0.25, 1.50]
        randomize_base_mass = False
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.    
  
    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.46
        class scales(LeggedRobotCfg.rewards.scales):
            torques = -0.0002
            dof_pos_limits = -5.0  # -10.0
            termination = -0.0
            tracking_lin_vel = 4.0  # 2.0
            tracking_ang_vel = 2.0  # 1.0
            lin_vel_z = -5.0  # -2.0
            ang_vel_xy = -0.5  # -0.05
            orientation = -0.1  # 0.
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0.5   # 0.
            feet_air_time = 0.1  # 1.0
            collision = -1.
            feet_stumble = -0.0 
            action_rate = -0.01
            stand_still = -0.
    
    class normalization(LeggedRobotCfg.normalization):
        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 0.1  # 100.
        clip_frequency = 2.0  # 1.0
        clip_step_height = 0.1  # 0.05

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0  # scales other values
        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            dof_pos = 0.01
            dof_vel = 0.5  # 1.5
            lin_vel = 0.1
            ang_vel = 0.1  # 0.2
            gravity = 0.05
            height_measurements = 0.1

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'  # none, plane, heightfield or trimesh
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 25  # [m]
        curriculum = False  # False
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = False
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.,
                             0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        # 1m x 1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.,
                             0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False  # select a unique terrain type and pass all arguments
        rough = True
        terrain_kwargs = None  # Dict of arguments for selected terrain
        max_init_terrain_level = 5  # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2] #[0.1, 0.1, 0.35, 0.25, 0.2, 0.2, 0.2]
        # trimesh only:
        slope_threshold = 0.  # slopes above this threshold will be corrected to vertical surfaces
    
    class sim(LeggedRobotCfg.sim):
        dt = 0.005

class STOCH3ForceControlCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24  # per iteration
        max_iterations = 3000  # number of policy updates

        # logging
        save_interval = 100  # check for potential saves every this many iterations
        run_name = 'exp1'
        experiment_name = 'stoch3fc'

        resume = False
        load_run = -1  # "Sep04_15-20-58_" # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and checkpoint
  
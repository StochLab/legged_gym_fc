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

class GO1Wrench8Cfg(LeggedRobotCfg):
    class env( LeggedRobotCfg.env):
        num_envs = 1024
        num_observations = 56 + 4
        num_privileged_obs = None
        num_partial_observations = 46
        history_size = 30
        # if not None a privileged_obs_buf will be returned by step()
        # (critic obs for asymmetric training). None is returned otherwise
        num_actions = 6 # 12
        num_joints = 12
        swing_height = 0.12
        step_frequency = 2.0
        stance_pc_factor = 1.2
        walking_height = 0.35
        env_spacing = 3.  # not used with height-fields/tri-meshes
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 20  # episode length in seconds
        evaluation = False

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.38]  # x,y,z [m]
        default_joint_angles = {'FL_hip_joint': 0.145, 'RL_hip_joint': 0.145,
                                'FR_hip_joint': -0.145, 'RR_hip_joint': -0.145,
                                'FL_thigh_joint': 0.75, 'RL_thigh_joint': 0.75,
                                'FR_thigh_joint': 0.75, 'RR_thigh_joint': 0.75,
                                'FL_calf_joint': -1.45, 'RL_calf_joint': -1.45,
                                'FR_calf_joint': -1.45, 'RR_calf_joint': -1.45}

    class observation:
        phase = True  # 4
        actual_linear_velocity = False  # 3
        estimated_linear_velocity = True #3
        angular_velocity = True  # 3
        orientation = True  # 3
        commands = True  # 3
        foot_location = True  # 12
        foot_contact = True  # 4
        dof_pos = True  # 12
        dof_vel = True  # 12
        desired_contacts = True

        class partial:
            angular_velocity = True  # 3
            orientation = True  # 3
            commands = False  # 3
            foot_location = True  # 12
            foot_contact = True  # 4
            dof_pos = True  # 12
            dof_vel = True  # 12

    class normalization( LeggedRobotCfg.normalization ):
        class obs_scales( LeggedRobotCfg.normalization.obs_scales ):
            lin_vel = 1.0
            ang_vel = 1.0
            dof_pos = 1.0
            dof_vel = 0.05
            foot_pos = 1.0
            phase = 0.30
            foot_contact = 1.0
            contact_force = 1.0
            height_measurements = 5.0

    class noise( LeggedRobotCfg.noise ):
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales( LeggedRobotCfg.noise.noise_scales ):
            dof_pos = 0.01
            dof_vel = 1.0
            foot_pos = 0.01
            phase = 0.01
            contact_force = 2.0  # 0.5
            foot_contact = 0.0
            lin_vel = 0.01
            ang_vel = 0.01
            gravity = 0.05
            height_measurements = 0.1

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 3 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 5. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-0.8, 0.8]   # min max [m/s]
            ang_vel_yaw = [-1.5, 1.5]    # min max [rad/s]
            heading = [-3.14, 3.14]


    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'T'
        controller = 'custom'

        if control_type == 'P' and controller == 'isaac':
            stiffness = {'joint': 400, 'cartesian': 0}  # [N*m/rad]
            damping = {'joint': 10, 'cartesian': 0}  # [N*m*s/rad]
            decimation = 4

        if control_type == 'P' and controller == 'custom':
            stiffness = {'joint': 40.0, 'cartesian': 0.0}
            damping = {'joint': 1.0, 'cartesian': 0.0}
            decimation = 2

        if control_type == 'T':
            stiffness = {'joint': 40.0, 'cartesian': 0.0}  # [N*m/rad]
            damping = {'joint': 1.0, 'cartesian': 0.0}  # [N*m*s/rad]
            decimation = 2 # 4

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.1
        # decimation: Number of control action updates @ sim DT per policy DT
        use_actuator_network = False
        actuator_net_file = "{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/unitree_go1.pt"

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1.urdf'
        name = "go1"
        foot_name = "calf" #"foot"
        penalize_contacts_on = ["thigh", "hip"] #["thigh", "hip", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = True  # Some .obj meshes must be flipped from y-up to z-up
        default_dof_drive_mode = 3
        # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        fix_base_link = False
        collapse_fixed_joints = True
        # merge bodies connected by fixed joints.
        # Specific fixed joints can be kept by adding " <... dont_collapse="true">
        # fix the base of the robot

    class domain_rand( LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 10
        max_push_vel_xy = 1.


    class rewards(LeggedRobotCfg.rewards):
        swing_height = 0.12
        step_frequency = 2.0
        walking_height = 0.35
        stance_pc_factor = 1.2
        soft_dof_pos_limit = 0.9
        base_height_target = 0.35
        max_contact_force = 90.0
        force_sigma = 50.
        vel_sigma = 0.5
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        class scales(LeggedRobotCfg.rewards.scales):
            torques = -0.0000
            dof_pos_limits = -10.0
            termination = -0.0
            tracking_lin_vel = 0.0
            tracking_lin_vel_x = 1.0
            tracking_lin_vel_y = 1.0
            tracking_ang_vel = 1.0
            tracking_dof_pos = 0.5
            lin_vel_z = -2.0    
            ang_vel_xy = -0.1
            orientation = -0.3
            dof_vel = -0.
            dof_acc = 0.0 # -2.5e-7
            base_height = -0.5  #-0.
            feet_air_time = 0.1
            collision = -1.
            symmetry = 0.5
            action_rate = -0.000025
            fx = 0.01
            fy = 0.01
            fz = 0.02
            tau_x = 0.01
            tau_y = 0.01
            tracking_contacts_shaped_force = 0.01
            tracking_contacts_shaped_vel = 0.01
            feet_slip = -0.001
            feet_impact_vel = -0.005
            feet_stumble = -0.25
            stand_still = -0.
            feet_forces = 0.00 # 0
            wrench_limits = -0.000


    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane' # 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        rough = True
        measure_heights = False
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        max_init_terrain_level = 5  # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.8, 0.2, 0.0, 0.0, 0.0]
        # trimesh only:
        slope_threshold = 0.0  # slopes above this threshold will be corrected to vertical surfaces

    class sim(LeggedRobotCfg.sim):
        dt = 0.005  # control timestep
        substeps = 2  # physics simulation timestep

        class physx:
            solver_type = 1
            num_position_iterations = 6  # 4 improve solver convergence
            num_velocity_iterations = 1  # keep default
            # shapes whose distance is less than the sum of their contactOffset values will generate contacts
            contact_offset = 0.02  # 0.02
            # two shapes will come to rest at a distance equal to the sum of their restOffset values
            rest_offset = 0.0
            # A contact with a relative velocity below this will not bounce.
            bounce_threshold_velocity = 0.2
            # The maximum velocity permitted to be introduced by the solver to correct for penetrations in contacts.
            max_depenetration_velocity = 100.0

class GO1Wrench8CfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 # 5.e-4
        schedule = 'adaptive'  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95 #0.95
        desired_kl = 0.005 #0.01
        max_grad_norm = 1.

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'go1_wrench8'
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24  # per iteration
        max_iterations = 24000  # number of policy updates
        # logging
        save_interval = 100  # check for potential saves every this many iterations
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and checkpoint
    class policy( LeggedRobotCfgPPO.policy):
        latent = False
        latent_dim = 80
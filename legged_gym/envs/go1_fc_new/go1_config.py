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

class GO1NewCfg(LeggedRobotCfg):
    class env( LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 76
        num_privileged_obs = None
        # if not None a priviledge_obs_buf will be returned by step()
        # (critic obs for assymetric training). None is returned otherwise
        num_actions = 12  # 12
        num_joints = 12
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 20  # episode length in seconds
        evaluation = False

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.33]  # x,y,z [m]
        default_joint_angles = {'FL_hip_joint': 0.0, 'RL_hip_joint': 0.0,
                                'FR_hip_joint': 0.0, 'RR_hip_joint': 0.0,
                                'FL_thigh_joint': 0.8, 'RL_thigh_joint': 0.8,
                                'FR_thigh_joint': 0.8, 'RR_thigh_joint': 0.8,
                                'FL_calf_joint': -1.6, 'RL_calf_joint': -1.6,
                                'FR_calf_joint': -1.6, 'RR_calf_joint': -1.6}

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        controller = 'custom'

        if control_type == 'P' and controller == 'isaac':
            stiffness = {'joint': 400, 'cartesian': 0}  # [N*m/rad]
            damping = {'joint': 10, 'cartesian': 0}  # [N*m*s/rad]
            decimation = 1

        if control_type == 'P' and controller == 'custom':
            stiffness = {'joint': 20.0, 'cartesian': 0.0}
            damping = {'joint': 1.0, 'cartesian': 0.0}
            decimation = 4

        if control_type == 'T':
            stiffness = {'joint': 20.0, 'cartesian': 0.0}  # [N*m/rad]
            damping = {'joint': 1.0, 'cartesian': 0.0}  # [N*m*s/rad]
            decimation = 4 # 4

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.1
        # decimation: Number of control action updates @ sim DT per policy DT
        # decimation: Number of control action updates @ sim DT per policy DT
        use_actuator_network = False
        actuator_net_file = "{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/unitree_go1.pt"

    class asset( LeggedRobotCfg.asset ):
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1_description/urdf/go1.urdf'
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1_working_2.urdf'
        name = "go1"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "hip", "calf"]
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
        push_interval_s = 15
        max_push_vel_xy = 1.


    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.30
        class scales(LeggedRobotCfg.rewards.scales):
            torques = -0.0002
            dof_pos_limits = -10.0
            termination = -0.0
            tracking_lin_vel = 2.0
            tracking_ang_vel = 1.0
            lin_vel_z = -2.0    
            ang_vel_xy = -0.05
            orientation = -0.1  #-0.
            dof_vel = -0.
            dof_acc = 0.0 # -2.5e-7
            base_height = -0.5  #-0.
            feet_air_time = 0.1
            collision = -1.
            feet_stumble = -0.25
            action_rate = -0.01
            stand_still = -0.
            angle_perturbation = 0

    class normalization( LeggedRobotCfg.normalization ):
        class obs_scales( LeggedRobotCfg.normalization.obs_scales ):
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            foot_pos = 1
            contact_force = 1
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 0.2  # 100.
        clip_frequency = 2.0  # 1.0
        clip_step_height = 0.1  #0.05

    class noise( LeggedRobotCfg.noise ):
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales( LeggedRobotCfg.noise.noise_scales ):
            dof_pos = 0.01
            dof_vel = 1.5
            foot_pos = 0.02
            contact_force = 2.0  # 0.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1


    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = False
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
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:
        slope_treshold = 0.0 # slopes above this threshold will be corrected to vertical surfaces

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

class GO1NewCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'go1_fc_new'
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24  # per iteration
        max_iterations = 12000  # number of policy updates
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
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

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
from ml_logger import logger as LOGGER

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, task_registry, Logger, class_to_dict
from legged_gym.utils.helpers import update_class_from_dict, export_as_jit

import numpy as np
import torch
import pickle


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    load_original_config = True
    if(load_original_config):
        try:
            config_path = args.load_run + '/config.pkl'
            with open(config_path,'rb') as file:
                env_cfg_dict = pickle.load(file)
                # print(env_cfg_dict)
                print("Config loaded successfully from:",config_path)
            update_class_from_dict(env_cfg, env_cfg_dict)
        except:
            print("Configuration file not found")
    else:
        print("Default Configuration loaded")
    # print("cfg type",type(env_cfg))
    # print("Check:",env_cfg.asset.default_dof_drive_mode)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    # print("Using Libtraj Max:", env_cfg.env.use_libtraj_max)
    train_cfg.policy.latent = False
    env_cfg.terrain.mesh_type='plane'
    env_cfg.env.evaluation = True
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.rough = True
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    obs_history = env.get_observations_history()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    #print("step1")
    policy = ppo_runner.get_inference_policy(device=env.device)
    env.add_velocity_estimator(ppo_runner.alg.velocity_model)
    #print("policy", ppo_runner.alg.actor_critic.actor)
    #torch.save(ppo_runner.alg.actor_critic.actor,"./policy.pth")

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:

        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_as_jit(ppo_runner.alg.actor_critic.actor, path, 'policy_'+args.run_name)
        print('Exported policy as jit script to: ', path)
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'velocity_estimator')
        print("Velocity Model",ppo_runner.alg.velocity_model.latent_model)
        export_as_jit(ppo_runner.alg.velocity_model.latent_model, path, 'velocity_'+args.run_name)

    LOGGER.log_params(Cfg=class_to_dict(env_cfg))
    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 500#100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    for i in range(10*int(env.max_episode_length)):
        actions = policy(obs.detach(),obs_history)
        obs, obs_history,_, _, rews, dones, infos = env.step(actions.detach())
        obs[:,13:16] = torch.tensor(env_cfg.env.num_envs*[[0.0,0.0,0.9]],dtype=torch.float32)
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            # logger.log_states(
            #     {
            #         'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
            #         'dof_pos': env.dof_pos[robot_index, joint_index].item(),
            #         'dof_vel': env.dof_vel[robot_index, joint_index].item(),
            #         'dof_torque': env.torques[robot_index, joint_index].item(),
            #         'command_x': env.commands[robot_index, 0].item(),
            #         'command_y': env.commands[robot_index, 1].item(),
            #         'command_yaw': env.commands[robot_index, 2].item(),
            #         'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
            #         'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
            #         'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
            #         'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
            #         'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
            #     }
            # )

            logger.log_states(
                {
                    'command_x': env.obs_buf[robot_index, 13].item(),
                    'command_y': env.obs_buf[robot_index, 14].item(),
                    'command_yaw': env.obs_buf[robot_index, 15].item(),
                    'base_vel_x': env.obs_buf[robot_index, 4].item(),
                    'base_vel_y': env.obs_buf[robot_index, 5].item(),
                    'base_vel_z': env.obs_buf[robot_index, 6].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'base_vel_x_actual': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y_actual': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z_actual': env.base_lin_vel[robot_index, 2].item(),
                    # 'foot_contact_1':(env.contact_forces[robot_index, env.feet_indices,2] > 1.0)[0].item(),
                    # 'foot_contact_2':(env.contact_forces[robot_index, env.feet_indices,2] > 1.0)[1].item(),
                    # 'foot_contact_3':(env.contact_forces[robot_index, env.feet_indices,2] > 1.0)[2].item(),
                    # 'foot_contact_4':(env.contact_forces[robot_index, env.feet_indices,2] > 1.0)[3].item(),
                    'orientation_x': env.projected_gravity[robot_index,0].item(),
                    'orientation_y': env.projected_gravity[robot_index,1].item(),
                    'orientation_z': env.projected_gravity[robot_index,2].item(),
                }
            )


            # logger.log_states(
            #     {
            #         'BL_x': env.foot_position[0,0].item(),
            #         'BR_x': env.foot_position[0,3].item(),
            #         'FL_x': env.foot_position[0,6].item(),
            #         'FR_x': env.foot_position[0,9].item(),
            #         'BL_y': env.foot_position[0,1].item(),
            #         'BR_y': env.foot_position[0,4].item(),
            #         'FL_y': env.foot_position[0,7].item(),
            #         'FR_y': env.foot_position[0,10].item(),
            #         'BL_z': env.foot_position[0,2].item(),
            #         'BR_z': env.foot_position[0,5].item(),
            #         'FL_z': env.foot_position[0,8].item(),
            #         'FR_z': env.foot_position[0,11].item()
            #     }
            # )

            # logger.log_states(
            #     {
            #         'BL_abd': env.dof_pos[0,0].item(),
            #         'BR_abd': env.dof_pos[0,3].item(),
            #         'FL_abd': env.dof_pos[0,6].item(),
            #         'FR_abd': env.dof_pos[0,9].item(),
            #         'BL_hip': env.dof_pos[0,1].item(),
            #         'BR_hip': env.dof_pos[0,4].item(),
            #         'FL_hip': env.dof_pos[0,7].item(),
            #         'FR_hip': env.dof_pos[0,10].item(),
            #         'BL_knee': env.dof_pos[0,2].item(),
            #         'BR_knee': env.dof_pos[0,5].item(),
            #         'FL_knee': env.dof_pos[0,8].item(),
            #         'FR_knee': env.dof_pos[0,11].item()
            #     }
            # )

        elif i==stop_state_log:
            logger.plot_states(plot_type='custom')
            # logger.plot_new_states()
            print("Logger called")
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
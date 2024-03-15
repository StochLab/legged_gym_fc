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
from legged_gym.utils import get_args, export_policy_as_jit, task_registry_bc, class_to_dict
from legged_gym.utils.logger2 import Logger
from legged_gym.utils.helpers import update_class_from_dict

import numpy as np
import torch
import pickle

def play(args):
    env_cfg, train_cfg = task_registry_bc.get_cfgs(name=args.task)
    # override some parameters for testing
    load_original_config = False
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
        env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
        train_cfg.policy.latent = False
        env_cfg.terrain.mesh_type = 'plane'  # 'trimesh', 'plane'
        env_cfg.env.evaluation = True
        env_cfg.terrain.num_rows = 5
        env_cfg.terrain.num_cols = 5
        env_cfg.terrain.curriculum = False
        env_cfg.terrain.rough = True
        env_cfg.noise.add_noise = False
        env_cfg.domain_rand.randomize_friction = False
        env_cfg.domain_rand.push_robots = False

        # prepare environment
        env, _ = task_registry_bc.make_env(name=args.task, args=args, env_cfg=env_cfg)
        obs = env.get_observations()
        # load policy
        train_cfg.runner.resume = True
        ppo_runner, train_cfg = task_registry_bc.make_alg_runner(env=env, name=args.task,
                                                              args=args, train_cfg=train_cfg)
        policy = ppo_runner.get_inference_policy(device=env.device)

        # export policy as a jit module (used to run it from C++)
        if EXPORT_POLICY:
            path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
            export_policy_as_jit(ppo_runner.alg.actor_critic, path)
            print('Exported policy as jit script to: ', path)

        LOGGER.log_params(Cfg=class_to_dict(env_cfg))
        logger = Logger(env.dt)
        robot_index = 0  # which robot is used for logging
        stop_state_log = 1000  # 100 # number of steps before plotting states
        stop_rew_log = env.max_episode_length + 1  # number of steps before print average episode rewards
        camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
        camera_vel = np.array([1., 1., 0.])
        camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
        img_idx = 0

        for i in range(10 * int(env.max_episode_length)):
            actions = policy(obs.detach())
            obs, _, _, rews, dones, infos = env.step(actions.detach())
            cmds = [0.0, 0.5, 0.]
            obs[:, 13:16] = torch.tensor(env_cfg.env.num_envs * [cmds], dtype=torch.float32)
            if RECORD_FRAMES:
                if i % 2:
                    filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported',
                                            'frames', f"{img_idx}.png")
                    env.gym.write_viewer_image_to_file(env.viewer, filename)
                    img_idx += 1
            if MOVE_CAMERA:
                camera_position += camera_vel * env.dt
                env.set_camera(camera_position, camera_position + camera_direction)

            if i < stop_state_log:
                logger.log_states(
                    {
                        'command_x': cmds[0],
                        'command_y': cmds[1],
                        'command_yaw': cmds[2],
                        'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                        'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                        'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                        'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                        'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy(),
                        'grfs': env.forces[robot_index, ...].cpu().numpy().reshape(4, 3),
                        'torques': env.torques[robot_index, :].cpu().numpy(),
                        'com_torques': env.com_torques[robot_index, ...].cpu().numpy()
                        # 'wrench': env.wrench[robot_index, :].cpu().numpy()
                    }
                )

            elif i == stop_state_log:
                print('writing the log to the desk')
                logger.write_states()
                print("Logger called")
            if 0 < i < stop_rew_log:
                if infos["episode"]:
                    num_episodes = torch.sum(env.reset_buf).item()
                    if num_episodes > 0:
                        logger.log_rewards(infos["episode"], num_episodes)
            elif i == stop_rew_log:
                logger.print_rewards()

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)


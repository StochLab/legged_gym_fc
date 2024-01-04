import time
import isaacgym
import torch
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

def display(env_cfg):
    print('--------------------- Environment configuration --------------------------')
    print('num_envs: ', env_cfg.env.num_envs)
    print('num_observations', env_cfg.env.num_observations)
    print('num_actions', env_cfg.env.num_actions)
    print('episode_length (in sec)', env_cfg.env.episode_length_s)
    print('------------------------ Control Configuration ---------------------------')
    print('control time step: ', env_cfg.sim.dt)
    print('Kp: ', env_cfg.control.stiffness['joint'])
    print('Kd: ', env_cfg.control.damping['joint'])
    print('decimation: ', env_cfg.control.decimation)
    print('-------------------------- Commands configuration -------------------------')
    print('num_commands: ', env_cfg.commands.num_commands)
    print('Heading command: ', env_cfg.commands.heading_command)
    print('lin_vel_x range: ', env_cfg.commands.ranges.lin_vel_x)
    print('lin_vel_y range: ', env_cfg.commands.ranges.lin_vel_y)
    print('ang_vel_yaw range: ', env_cfg.commands.ranges.ang_vel_yaw)
    if env_cfg.commands.heading_command:
        print('heading range: ', env_cfg.commands.ranges.heading)
    print('----------------------- Initial state of the robot -------------------------')
    print('init pos: ', env_cfg.init_state.pos)
    print('init orientation: ', env_cfg.init_state.rot)
    print('init lin vel: ', env_cfg.init_state.lin_vel)
    print('init ang vel: ', env_cfg.init_state.ang_vel)
    print('init joint angles: ')
    for key, value in env_cfg.init_state.default_joint_angles.items():
        print(key, value)
    print('------------------------- Terrain configuration -----------------------------')
    print('mesh type: ', env_cfg.terrain.mesh_type)
    print('curriculum: ', env_cfg.terrain.curriculum)
    print('randomise friction: ', env_cfg.domain_rand.randomize_friction)
    print('static friction: ', env_cfg.terrain.static_friction)
    print('dynamic friction: ', env_cfg.terrain.dynamic_friction)
    if env_cfg.domain_rand.randomize_friction:
        print('friction range: ', env_cfg.domain_rand.friction_range)
    print('-----------------------------------------------------------------------------')

def train(args):

    # To see the available methods in gym
    # gym = isaacgym.gymapi.acquire_gym()
    # print(dir(gym))

    # Uses the command line args to overwrite the default args and sets up the environment
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    display(env_cfg)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    ppo_runner.learn(config=env_cfg,
                     num_learning_iterations=train_cfg.runner.max_iterations,
                     init_at_random_ep_len=True)


if __name__ == '__main__':
    args = get_args()

    # If task == stoch3lib, then
    # task_class = legged_gym.envs.stoch3lib.legged_robot_libtraj.LeggedRobotLibTraj
    train(args)



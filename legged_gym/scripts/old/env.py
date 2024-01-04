from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym.envs.base.legged_robot import LeggedRobot
from isaacgym import gymapi
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, parse_sim_params
from legged_gym import LEGGED_GYM_ROOT_DIR 
import os
import numpy as np
import torch

gym = gymapi.acquire_gym()
sim_device_id = 0
graphics_device_id = 0
physics_engine =  gymapi.SIM_PHYSX
cfg, _ = task_registry.get_cfgs('stoch3')
args = get_args()
sim_params = {"sim": class_to_dict(cfg.sim)}
sim_params = parse_sim_params(args, sim_params)

print(" Sim param dt:",sim_params.dt)

sim = gym.create_sim(sim_device_id, graphics_device_id, physics_engine, sim_params)



asset_path = cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
asset_root = os.path.dirname(asset_path)
asset_file = os.path.basename(asset_path)

asset_options = gymapi.AssetOptions()
asset_options.default_dof_drive_mode = cfg.asset.default_dof_drive_mode
asset_options.collapse_fixed_joints = cfg.asset.collapse_fixed_joints
asset_options.replace_cylinder_with_capsule = cfg.asset.replace_cylinder_with_capsule
asset_options.flip_visual_attachments = cfg.asset.flip_visual_attachments
asset_options.fix_base_link = cfg.asset.fix_base_link
asset_options.density = cfg.asset.density
asset_options.angular_damping = cfg.asset.angular_damping
asset_options.linear_damping = cfg.asset.linear_damping
asset_options.max_angular_velocity = cfg.asset.max_angular_velocity
asset_options.max_linear_velocity = cfg.asset.max_linear_velocity
asset_options.armature = cfg.asset.armature
asset_options.thickness = cfg.asset.thickness
asset_options.disable_gravity = cfg.asset.disable_gravity

robot = gym.load_asset(sim, asset_root, asset_file, asset_options)
dof_props_asset = gym.get_asset_dof_properties(robot)

print("number of joints",gym.get_asset_joint_count(robot))
#print("dof props asset", gym.get_asset_dof_dict(robot))
print("dof asset names", gym.get_asset_dof_names(robot))
print("rigid body asset names", gym.get_asset_rigid_body_names(robot))

# env = LeggedRobot(  cfg=cfg,
#                     sim_params=sim_params,
#                     physics_engine=args.physics_engine,
#                     sim_device=args.sim_device,
#                     headless=args.headless)

# count = 0
# env.reset()
# while(count<1000):
#     temp = env.step(torch.tensor(np.random.uniform(-1,1,(1,12)),dtype=torch.float32))
#     count+=1
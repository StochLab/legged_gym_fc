import numpy as np
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from go1_utils import *
import math

controller_dt = 0.01  # in sec

gym = gymapi.acquire_gym()
sim = acquire_sim(gym, controller_dt)
add_ground(gym, sim)

# set up the env grid
num_envs = 1
envs_per_row = int(math.sqrt(num_envs))
env_spacing = 0.5

# one actor per env
envs, actors = create_envs(gym, sim, num_envs, envs_per_row, env_spacing)
# force_sensors = get_force_sensor(gym, envs, actors)
cam_pos = gymapi.Vec3(2, 2, 2) # w.r.t target env
viewer = add_viewer(gym, sim, envs[0], cam_pos)

for idx in range(num_envs):
    # configure the joints for effort control mode (once)
    props = gym.get_actor_dof_properties(envs[idx], actors[idx])
    props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
    props["stiffness"].fill(0.0)
    props["damping"].fill(0.0)
    gym.set_actor_dof_properties(envs[idx], actors[idx], props)

count = 1
render_fps = 30
render_count = int(1/render_fps/controller_dt)

# simulation loop
while not gym.query_viewer_has_closed(viewer):
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # current_time = gym.get_sim_time(sim)
    commands = np.zeros(3, dtype=DTYPE)
    commands[0] = 0.1
    commands[2] = 0.0

    # run controllers
    for idx, (env, actor, controller) in enumerate(zip(envs, actors, controllers)):
        dof_states = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
        body_idx = gym.find_actor_rigid_body_index(env, actor, controller._quadruped._bodyName, gymapi.DOMAIN_ACTOR)
        body_states = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_ALL)[body_idx]
        legTorques = controller.run(dof_states, body_states, commands)
        gym.apply_actor_dof_efforts(env, actor, legTorques / (controller_dt * 100))

    if count % render_count == 0:
        # update the viewer
        count = 0
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.clear_lines(viewer)

    # Wait for dt to elapse in real time.
    gym.sync_frame_time(sim)
    count += 1

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
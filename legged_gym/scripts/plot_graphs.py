from matplotlib import pyplot as plt
import pickle as pkl
import numpy as np


def plot_velocities(data, start, stop):
    nb_rows = 2
    nb_cols = 2
    fig, axs = plt.subplots(nb_rows, nb_cols)
    dt = 0.01
    L = len(data['command_x'])

    if start < 0:
        start = 0
    if stop > L-1:
        stop = L-1

    N = stop - start
    time = np.linspace(0, N * dt, N)

    axs[0, 0].plot(time, data['command_x'][start:stop], label='target')
    axs[0, 0].plot(time, data['base_vel_x'][start:stop],  'r', label='measured')
    axs[0, 0].set(xlabel='time [s]', ylabel='velocity (m/s)', title='forward velocity')
    axs[0, 0].legend()

    axs[0, 1].plot(time, data['command_y'][start:stop], label='target')
    axs[0, 1].plot(time, data['base_vel_y'][start:stop], 'r', label='measured')
    axs[0, 1].set(xlabel='time [s]', ylabel='velocity (m/s)', title='lateral velocity')
    axs[0, 1].legend()

    axs[1, 0].plot(time, time * 0, label='target')
    axs[1, 0].plot(time, data['base_vel_y'][start:stop],  'r', label='measured')
    axs[1, 0].set(xlabel='time [s]', ylabel='velocity (m/s)', title='z-velocity')
    axs[1, 0].legend()

    axs[1, 1].plot(time, data['command_yaw'][start:stop], label='target')
    axs[1, 1].plot(time, data['base_vel_yaw'][start:stop],  'r', label='measured')
    axs[1, 1].set(xlabel='time [s]', ylabel='velocity (rad/s)', title='yaw velocity')
    axs[1, 1].legend()

    plt.show()

def plot_com_torques(data, start, stop):
    nb_rows = 3
    nb_cols = 1
    fig, axs = plt.subplots(nb_rows, nb_cols)
    dt = 0.01
    torques = np.array(data['com_torques'])
    L = len(torques)

    if start < 0:
        start = 0
    if stop > L - 1:
        stop = L - 1

    N = stop - start
    time = np.linspace(0, N * dt, N)

    axs[0].plot(time, torques[start:stop, 0], label='tau_x')
    axs[0].set(xlabel='time [s]', ylabel='torque (Nm)')
    axs[0].legend()

    axs[1].plot(time, torques[start:stop, 1], label='tau_y')
    axs[1].set(xlabel='time [s]', ylabel='torque (Nm)')
    axs[1].legend()

    axs[2].plot(time, torques[start:stop, 1], label='tau_z')
    axs[2].set(xlabel='time [s]', ylabel='torque (Nm)')
    axs[2].legend()

    fig.suptitle('Torques about com')

    plt.show()

def plot_grfs(data, start, stop, axis='z'):
    nb_rows = 4
    nb_cols = 1
    fig, axs = plt.subplots(nb_rows, nb_cols)
    dt = 0.01
    forces = np.array(data['grfs'])
    L = len(forces)

    map = dict()
    map['x'] = 0
    map['y'] = 1
    map['z'] = 2

    if start < 0:
        start = 0
    if stop > L - 1:
        stop = L - 1

    N = stop - start
    time = np.linspace(0, N * dt, N)

    fl_forces = forces[:, 0, map[axis]].tolist()
    fr_forces = forces[:, 1, map[axis]].tolist()
    bl_forces = forces[:, 2, map[axis]].tolist()
    br_forces = forces[:, 3, map[axis]].tolist()

    axs[0].plot(time, fl_forces[start:stop], label='FL')
    axs[0].set(xlabel='time [s]', ylabel='force (N)')
    axs[0].legend()

    axs[1].plot(time, fr_forces[start:stop], 'r', label='FR')
    axs[1].set(xlabel='time [s]', ylabel='force (N)')
    axs[1].legend()

    axs[2].plot(time, bl_forces[start:stop], 'r', label='BL')
    axs[2].set(xlabel='time [s]', ylabel='force (N)')
    axs[2].legend()

    axs[3].plot(time, br_forces[start:stop], 'r', label='BR')
    axs[3].set(xlabel='time [s]', ylabel='force (N)')
    axs[3].legend()

    fig.suptitle('Commanded forces in ' + axis + ' direction')

    plt.show()

def plot_torques(data, start, stop):
    nb_rows = 3
    nb_cols = 4
    fig, axs = plt.subplots(nb_rows, nb_cols)
    dt = 0.01
    torques = np.array(data['torques'])
    L = len(torques)

    if start < 0:
        start = 0
    if stop > L - 1:
        stop = L - 1
    N = stop - start
    time = np.linspace(0, N * dt, N)

    legs = ['FL', 'FR', 'BL', 'BR']
    joints = ['abd', 'hip', 'knee']
    colors = ['r', 'b', 'g']

    for i in range(12):
        leg = i // 3
        joint = i % 3
        axs[joint, leg].plot(time, torques[:, i].tolist()[start:stop],
                             colors[joint], label=joints[joint])
        axs[joint, leg].legend()

    fig.suptitle('Commanded torques for FL, FR, BL, BR')
    plt.show()

def plot_dof_pos(data, start, stop):
    nb_rows = 3
    nb_cols = 4
    fig, axs = plt.subplots(nb_rows, nb_cols)
    dt = 0.01
    dof_pos = np.array(data['delta_dof_pos'])
    L = len(dof_pos)

    if start < 0:
        start = 0
    if stop > L - 1:
        stop = L - 1
    N = stop - start
    time = np.linspace(0, N * dt, N)

    legs = ['FL', 'FR', 'BL', 'BR']
    joints = ['abd', 'hip', 'knee']
    colors = ['r', 'b', 'g']

    for i in range(12):
        leg = i // 3
        joint = i % 3
        axs[joint, leg].plot(time, dof_pos[:, i].tolist()[start:stop],
                             colors[joint], label=legs[leg]+'_'+joints[joint])
        axs[joint, leg].legend()

    fig.suptitle('desired dof perturbations about joints')
    plt.show()








def plot_wrench(data, start, stop):
    nb_rows = 2
    nb_cols = 3
    fig, axs = plt.subplots(nb_rows, nb_cols)
    dt = 0.01
    wrench = np.array(data['wrench'])
    L = len(wrench)

    if start < 0:
        start = 0
    if stop > L - 1:
        stop = L - 1

    N = stop - start
    time = np.linspace(0, N * dt, N)

    F_x = wrench[:, 0].tolist()
    F_y = wrench[:, 1].tolist()
    F_z = wrench[:, 2].tolist()
    M_x = wrench[:, 3].tolist()
    M_y = wrench[:, 4].tolist()
    M_z = wrench[:, 5].tolist()

    axs[0, 0].plot(time, F_x[start:stop], label='$F_x$')
    axs[0, 0].set(xlabel='time [s]', ylabel='force (N)')
    axs[0, 0].legend()

    axs[0, 1].plot(time, F_y[start:stop], label='$F_y$')
    axs[0, 1].set(xlabel='time [s]', ylabel='force (N)')
    axs[0, 1].legend()

    axs[0, 2].plot(time, F_z[start:stop], label='$F_z$')
    axs[0, 2].set(xlabel='time [s]', ylabel='force (N)')
    axs[0, 2].legend()

    axs[1, 0].plot(time,    M_x[start:stop], label='$M_x$')
    axs[1, 0].set(xlabel='time [s]', ylabel='torque (Nm)')
    axs[1, 0].legend()

    axs[1, 1].plot(time, M_y[start:stop], label='$M_y$')
    axs[1, 1].set(xlabel='time [s]', ylabel='torque (Nm)')
    axs[1, 1].legend()

    axs[1, 2].plot(time, M_z[start:stop], label='$M_z$')
    axs[1, 2].set(xlabel='time [s]', ylabel='torque (Nm)')
    axs[1, 2].legend()

    fig.suptitle('wrench on COM')
    plt.show()


data = pkl.load(open('log_file.pkl', 'rb'))
start = 400
stop = 900
plot_velocities(data, start, stop)
plot_grfs(data, start, stop, axis='z')
plot_grfs(data, start, stop, axis='y')
plot_grfs(data, start, stop, axis='x')
plot_torques(data, start, stop)
# plot_com_torques(data, start, stop)
plot_wrench(data, start, stop)
plot_dof_pos(data, start, stop)






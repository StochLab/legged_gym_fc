# ### Trajectory Generator
# Written by Tejas Rane, Aditya Shirwatkar (July, 2021)
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Utilities for realizing walking controllers.

Action space is defined as: nx18 where n is the number of batches
  action[:, 0:4  ] -> x_shift: fl fr bl br
  action[:, 4:8  ] -> y_shift: fl fr bl br
  action[:, 8:12 ] -> z_shift: fl fr bl br
  action[:, 12:15] -> cmd_vel_lin: vx vy vz
  action[:, 15:18] -> cmd_vel_ang: wx wy wz
"""

from typing import NamedTuple, Tuple
from collections import namedtuple
from legged_gym.envs.go1_wrench6.kinematics import Stoch3Kinematics
# from utils.logger import DataLog
import numpy as np
import matplotlib.pyplot as plt
from fabulous.color import red

import torch as th

PI: float = np.pi
FL = 0
FR = 1
BL = 2
BR = 3


class LegData:
    def __init__(self, name: str, ID: int, frame: np.ndarray, Nb: int = 1):
        self.name = name
        self.ID = ID
        self.frame = frame
        self.Nb = Nb
        self.theta: float = 0.0
        self.true_is_stance: np.ndarray = np.zeros(Nb, dtype=bool)
        self.batch_prev_motor_angles: np.ndarray = np.zeros((Nb, 3), dtype=float)
        self.batch_curr_motor_angles: np.ndarray = np.zeros((Nb, 3), dtype=float)
        self.batch_stance_bool: np.ndarray = np.zeros((Nb, 1), dtype=bool)
        self.batch_prev_ee: np.ndarray = np.zeros((Nb, 3), dtype=float)
        self.batch_curr_ee: np.ndarray = np.zeros((Nb, 3), dtype=float)
        self.batch_shifts: np.ndarray = np.zeros((Nb, 3), dtype=float)
        self.batch_foot_vel: np.ndarray = np.zeros((Nb, 3), dtype=float)


class GaitData:
    """
        frequency: np.ndarray = 2.5,
        max_linear_xvel: np.ndarray = 2.5/5, max_linear_yvel: np.ndarray = 2.5/10, max_angular_vel: np.ndarray = 2.5,
        swing_height: np.ndarray = 0.060, torso_height: np.ndarray = -0.25,
        stance_start: np.ndarray = [0, PI, PI, 0], stance_duration: np.ndarray = [PI, PI, PI, PI]
    """

    def __init__(self, frequency: float = 2.5, swing_height: float = 0.08, torso_height: float = -0.45,
                 stance_start: np.ndarray = [0, PI, PI, 0], stance_duration: np.ndarray = [PI, PI, PI, PI]):
        """Creates trot as default"""
        self.frequency = frequency
        self.max_linear_xvel = 1.0
        self.max_linear_yvel = 1.0
        self.max_angular_vel = 1.0
        self.swing_height = swing_height
        self.torso_height = torso_height
        self.stance_start = stance_start
        self.stance_duration = stance_duration
        self.omega = self.frequency * 2 * PI


class TrajectoryGenerator():
    """
    A class that generates swing and stance trajectories for walking based on
    input velocity.
    Note: An assumption is made that the trajectory cycle theta is the same for all the batches
    """

    def __init__(self, device, batch_size=1, sim="isaac", gait_type='trot', frequency=2.5, torso_height=-0.45,
                 swing_height=0.15, link_parameters=[0.123, 0.297, 0.347], torso_dims=[0.541, 0.203, 0.1],
                 use_contact_info=False, stance_pc_factor=1.0):
        self._device = device
        self.Nb: int = batch_size
        self.theta = np.zeros((batch_size, 1))  # self.theta = 0.0
        self.use_contact_info = use_contact_info

        self.stoch3_kin = Stoch3Kinematics(link_parameters=link_parameters, torso_dims=torso_dims)
        self.robot_width = self.stoch3_kin.torso_dims[1]
        self.robot_length = self.stoch3_kin.torso_dims[0]
        self.link_lengths_stoch3 = np.array(self.stoch3_kin.link_lengths)

        self.front_left = LegData('FL', FL, self.stoch3_kin.leg_frames[0, ...].reshape((1, 3)), batch_size)
        self.front_right = LegData('FR', FR, self.stoch3_kin.leg_frames[1, ...].reshape((1, 3)), batch_size)
        self.back_left = LegData('BL', BL, self.stoch3_kin.leg_frames[2, ...].reshape((1, 3)), batch_size)
        self.back_right = LegData('BR', BR, self.stoch3_kin.leg_frames[3, ...].reshape((1, 3)), batch_size)

        if gait_type == 'trot':
            self.gait = GaitData(frequency=frequency,
                                 swing_height=swing_height, torso_height=torso_height,
                                 stance_start=[PI, 0, 0, PI],
                                 stance_duration=[stance_pc_factor* PI]*4)
        elif gait_type == 'crawl':
            self.gait = GaitData(frequency=0.5,
                                 swing_height=0.15, torso_height=-0.45,
                                 stance_start=[5 * PI / 3, 2 * PI / 3, 4 * PI / 3, 1 * PI / 3],
                                 stance_duration=[5.1 * PI / 3, 5.1 * PI / 3, 5.1 * PI / 3, 5.1 * PI / 3])
        else:
            print(red("Unknown Gait Type, selecting default trot gait"))
            self.gait = GaitData(frequency=2.5,
                                 swing_height=0.15, torso_height=-0.45,
                                 stance_start=[0, PI, PI, 0],
                                 stance_duration=[PI, PI, PI, PI])

        self.sim = sim

        self.stance_bool = th.tensor(
            np.concatenate([self.front_left.batch_stance_bool,
                            self.front_right.batch_stance_bool,
                            self.back_left.batch_stance_bool,
                            self.back_right.batch_stance_bool], axis=1), dtype=bool).to(device=self._device)


    def constrain_theta(self, theta: float) -> float:
        """
        A function to constrain theta between [0, 2*Pi]

        Args:
            theta : trajectory cycling parameter
        """
        theta = np.fmod(theta, 2 * PI)
        # if theta < 0:
        #     theta = theta + 2*PI
        theta = theta + (theta < 0) * (2 * PI * np.ones(theta.shape))
        return theta

    def update_leg_theta(self, dt: float):
        '''
        Function to calculate the leg cycles of the trajectory in batches, depending on the gait.

        Args:
            dt : time step
        '''
        self.theta: np.ndarray = self.constrain_theta(self.theta + self.gait.omega * dt)

    def reset_theta(self, idx):
        '''
        Function to reset the main cycle of the trajectory.
        '''
        self.theta[idx, 0] = 0.0

    def initialize_traj_shift(self, shifts: np.ndarray):
        '''
        Initialize desired X, Y, Z offsets of trajectory for each leg

        Args:
            shifts : Translational shifts as an Nd-array of size = (batch_size, num_legs, 3)
        '''
        self.front_left.batch_shifts = shifts[:, FL, :].reshape((self.Nb, 3))
        self.front_right.batch_shifts = shifts[:, FR, :].reshape((self.Nb, 3))
        self.back_left.batch_shifts = shifts[:, BL, :].reshape((self.Nb, 3))
        self.back_right.batch_shifts = shifts[:, BR, :].reshape((self.Nb, 3))

    def initialize_prev_motor_ang(self, prev_motor_angles: np.ndarray):
        '''
        Initialize motor angles of previous time-step for each leg

        Args:
            prev_motor_angles : Previous time step encoder recorded joint angles as an Nd-array of size = (batch_size, 3*num_legs)
        '''
        self.front_left.batch_prev_motor_angles = prev_motor_angles[:, 0:3].reshape((self.Nb, 3))
        self.front_right.batch_prev_motor_angles = prev_motor_angles[:, 3:6].reshape((self.Nb, 3))
        self.back_left.batch_prev_motor_angles = prev_motor_angles[:, 6:9].reshape((self.Nb, 3))
        self.back_right.batch_prev_motor_angles = prev_motor_angles[:, 9:12].reshape((self.Nb, 3))

    def foot_step_planner(self, leg: LegData, v_leg: np.ndarray) -> np.ndarray:
        '''
        Calculates the  absolute coordinate (wrt hip frame) where the foot should land at the beginning of the stance phase of the trajectory based on the
        commanded velocities (either from joystick or augmented by the policy).
        Args:
            leg   : the leg for which the trajectory has to be calculated
            v_leg : the velocity vector for the leg (summation of linear and angular velocity components), with shape = (batch_size, 3)
        Ret:
            s : absolute coordinate of the foot step, with shape = (batch_size, 3)
        '''
        stance_time: float = (self.gait.stance_duration[leg.ID] / (2 * PI)) * (1.0 / self.gait.frequency)

        # print("v leg", v_leg.shape)
        # print("stance time",stance_time.shape)
        # print("leg batch shift", leg.batch_shifts.shape)
        s: np.ndarray = v_leg * stance_time / 2 + leg.batch_shifts

        # print("s::",s)

        return s

    def getLegPhase(self, leg: LegData) -> float:
        leg_phase = self.theta - self.gait.stance_start[leg.ID]
        # if (leg_phase < 0):
        #     leg_phase += 2 * PI
        leg_phase = leg_phase + (leg_phase < 0) * (2 * PI * np.ones(leg_phase.shape))
        return leg_phase.reshape(-1, 1)

    def isStance(self, leg: LegData):
        leg_phase = self.getLegPhase(leg)
        isStance = leg_phase.copy()
        pc_complete = leg_phase / self.gait.stance_duration[leg.ID]
        indices = np.where(leg_phase <= self.gait.stance_duration[leg.ID])[0]
        thresh = 0.60 # 0.80
        if len(indices) > 0:
            sub_indices = (pc_complete[indices] > thresh) & self.use_contact_info
            isStance[indices[sub_indices[:, 0]], 0] = leg.true_is_stance[indices[sub_indices[:, 0]]]
            isStance[indices[~sub_indices[:, 0]], 0] = True
        pc_complete = (leg_phase - self.gait.stance_duration[leg.ID]) / (2 * PI - self.gait.stance_duration[leg.ID])
        indices = np.where(leg_phase > self.gait.stance_duration[leg.ID])[0]
        if len(indices) > 0:
            sub_indices = (pc_complete[indices] > thresh) & self.use_contact_info
            isStance[indices[sub_indices[:, 0]], 0] = leg.true_is_stance[indices[sub_indices[:, 0]]]
            isStance[indices[~sub_indices[:, 0]], 0] = False

        return isStance.reshape(-1, 1)

    def calculate_planar_traj(self, leg: LegData, aug_6D_twist: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Calculates the x and y component of the trajectory based on the commanded velocities (either from joystick or augmented by the policy).
        Args:
            leg          : the leg for which the trajectory has to be calculated
            aug_6D_twist : augmented body twist commanded by the policies, with shape = (batch_size, 6)
            dt           : control period
        Ret:
            r : calculated x and y coordinates of the trajectory in batches.
            v : calculated vx and vy linear of the trajectory in batches.
        '''

        # Since we will consider only the 2D motion of the robot, we will ignore the linear z component of the twist
        cmd_lvel = np.concatenate([aug_6D_twist[:, 0:2], np.zeros((self.Nb, 1))], axis=1)
        # Since we will consider only the 2D motion of the robot, we will consider the angular z component of the twist
        cmd_avel = np.concatenate([np.zeros((self.Nb, 2)), aug_6D_twist[:, 5].reshape(-1, 1)], axis=1)

        v_leg = np.zeros((self.Nb, 3), dtype=float)  # Only consider linear velocity of the foot
        next_step = np.zeros((self.Nb, 3), dtype=float)
        swing_vec = np.array((self.Nb, 3), dtype=float)

        prev_foot_pos = np.concatenate([leg.batch_prev_ee[:, 0:2],
                                        np.zeros((self.Nb, 1), dtype=float)], axis=1).reshape((self.Nb, 3))

        # print("leg.frame",leg.frame)
        # print("prev_foot_pos",type(prev_foot_pos))
        # prev_r = prev_foot_pos
        prev_r = prev_foot_pos + leg.frame

        v_lcomp = cmd_lvel
        v_acomp = np.cross(cmd_avel, prev_r)
        # if(leg.name == 'FL'):
        #   print("leg position",prev_r)
        v_leg = v_lcomp + v_acomp

        isStance_bool = self.isStance(leg)
        # if self.isStance(leg):
        #     flag = -1 #during stance_phase of walking, leg moves backwards to push body forward
        #     dr = v_leg * dt * flag
        #     r = prev_r + dr - leg.frame
        # else:
        #     flag = 1 #during swing_phase of walking, leg moves forward to next step
        #     next_step = self.foot_step_planner(leg, v_leg) + leg.frame
        #     swing_vec = next_step - prev_r
        #     leg_phase = self.getLegPhase(leg)
        #     time_left = (2 * PI - leg_phase) / (2 * PI) * (1 / self.gait.frequency)
        #     if time_left == 0:
        #         dr = 0
        #     else:
        #         dr = swing_vec/time_left * dt * flag
        #     r = prev_r + dr - leg.frame

        flag = -1 * isStance_bool + (isStance_bool == False)
        # if(leg.ID==0):
        #    print("Flag",flag)
        # print("v leg shape",v_leg.shape)
        # print("flag",flag.shape)
        # print("is stance bool", isStance_bool.shape)
        dr_1 = v_leg * dt * (flag * isStance_bool)
        r_1 = prev_r + dr_1 - leg.frame

        next_step = self.foot_step_planner(leg, v_leg) + leg.frame
        # print("next step", next_step.shape)
        swing_vec = next_step - prev_r
        # print("swing vec",swing_vec.shape)
        leg_phase = self.getLegPhase(leg)
        # print("leg phase",leg_phase.shape)
        time_left = (2 * PI - leg_phase) / (2 * PI) * (1 / self.gait.frequency)
        # print("time left",time_left.shape)
        dr_2 = (time_left > 1e-4) * swing_vec / time_left * dt * flag * (isStance_bool == False)
        r_2 = prev_r + dr_2 - leg.frame

        r = r_1 * (isStance_bool) + r_2 * (isStance_bool == False)

        # print("time left",time_left,"dr2",dr_2,"theta",self.theta)
        # print("next_step",next_step,"prev_r",prev_r)
        # if(leg.ID == 0):
        #   print("xy plane position",r[0,:2])

        return r[:, 0:2], flag * v_leg[:, 0:2]

    def cspline_coeff(self, z0: np.ndarray, z1: np.ndarray, d0: np.ndarray, d1: np.ndarray, t: float):
        '''
        Generates coefficients for the sections of the cubic spline based on the boundary conditions
        Equation -> z = coefft[3]*t**3 + coefft[2]*t**2 + coefft[1]*t**1 + coefft[0]*t**0
        Args:
            z0 : initial z, with shape = (batch_size, 1)
            z1 : final z, with shape = (batch_size, 1)
            d0 : initial dz/dtheta, with shape = (batch_size, 1)
            d1 : final dz/dtheta, with shape = (batch_size, 1)
            t  : domain of the section [(0, z0) to (t, z1), initial and final control points]
        Ret:
            coefft : list of coefficients for that section of cubic spline.
        '''

        coefft = np.zeros((self.Nb, 4), dtype=float)
        coefft[:, 0] = z0
        coefft[:, 1] = d0
        w0 = z1 - z0 - d0 * t
        w1 = d1 - d0
        coefft[:, 2] = -1 * (-3 * t ** 2 * w0 + t ** 3 * w1) / t ** 4
        coefft[:, 3] = -1 * (2 * t * w0 - t ** 2 * w1) / t ** 4
        return coefft

    def calculate_vert_comp(self, leg: LegData) -> Tuple[np.ndarray, np.ndarray, bool]:
        '''
        Calculates the z component of the trajectory. The function for the z component can be changed here.
        The z component calculation is kept independent as it is not affected by the velocity calculations.
        Various functions can be used to smoothen out the foot impacts while walking.
        Args:
            leg : the leg for which the trajectory has to be calculated
        Ret:
            z    : calculated z component of the trajectory in batches
            dz   : linear velocity component of the trajectory in batches
            flag : flag for the virtual (assumed) foot contact
        '''

        # if not self.isStance(leg): # theta taken from +x, CW # Flip this sigh if the trajectory is mirrored
        #     flag = 1
        # else:
        #     flag = 0

        flag = (self.isStance(leg) == False)

        swing_phase = self.getLegPhase(leg) - self.gait.stance_duration[leg.ID]
        swing_duration = 2 * PI - self.gait.stance_duration[leg.ID]
        theta_leg = PI * swing_phase / swing_duration
        dtheta_leg = PI * self.gait.omega / swing_duration

        # Sine function
        z: np.ndarray = self.gait.swing_height * np.sin(theta_leg) * \
                        flag * np.ones((self.Nb, 1)) + self.gait.torso_height
        # if(z[0][0]>-0.45):
        #    print("z",z,"leg",leg.name,"theta::",theta_leg)

        # dz/dt = dz/dtheta * dtheta/dt
        dz: np.ndarray = self.gait.swing_height * np.cos(theta_leg) * \
                         flag * np.ones((self.Nb, 1)) * dtheta_leg

        """CONVERSION TO BATCH FORM LEFT"""
        # Cubic Spline
        # '''
        # This cubic spline is defined by 5 control points (n=4). Each control point is (theta, z) ie (theta_0, z_0) to (theta_n, z_n) n+1 control points
        # The assumed cubic spline is of the type:
        # z = coefft_n[3]*(t-t_n)**3 + coefft_n[2]*(t-t_n)**2 + coefft_n[1]*(t-t_n)**1 + coefft_n[0]*(t-t_n)**0 {0<=n<=3}
        # where, n denotes each section of the cubic spline, governed by the nth-index control point
        # '''
        # # theta = [0, PI/4, PI/2, 3*PI/4, PI]
        # z = [0.0, 3*self.foot_clearance/4, self.foot_clearance, self.foot_clearance/2, 0.0]
        # d = [0.1, 0.05, 0.0, -0.1, 0.0] # dz/dtheta at each control point
        # t_vec = []
        # dt_vec = []
        # coeffts = []

        # if(leg.theta < PI/4):
        #     idx = 0
        #     coeffts = self.cspline_coeff(z[idx], z[idx+1], d[idx], d[idx+1], PI/4)
        #     t_vec = [leg.theta**i for i in range(4)]
        #     dt_vec = [0] + [i*leg.theta**(i-1) for i in range(1,4)]
        # elif(leg.theta >= PI/4 and leg.theta < PI/2):
        #     idx = 1
        #     coeffts = self.cspline_coeff(z[idx], z[idx+1], d[idx], d[idx+1], PI/4)
        #     t_vec = [(leg.theta - PI/4)**i for i in range(4)]
        #     dt_vec = [0] + [i*(leg.theta - PI/4)**(i-1) for i in range(1,4)]
        # elif(leg.theta >= PI/2 and leg.theta < 3*PI/4):
        #     idx = 2
        #     coeffts = self.cspline_coeff(z[idx], z[idx+1], d[idx], d[idx+1], PI/4)
        #     t_vec = [(leg.theta - 2*PI/4)**i for i in range(4)]
        #     dt_vec = [0] + [i*(leg.theta - 2*PI/4)**(i-1) for i in range(1,4)]
        # elif(leg.theta >= 3*PI/4 and leg.theta < PI):
        #     idx = 3
        #     coeffts = self.cspline_coeff(z[idx], z[idx+1], d[idx], d[idx+1], PI/4)
        #     t_vec = [(leg.theta - 3*PI/4)**i for i in range(4)]
        #     dt_vec = [0] + [i*(leg.theta - 3*PI/4)**(i-1) for i in range(1,4)]
        # t_vec = np.array(t_vec)
        # coeffts = np.array(coeffts)
        # val = coeffts.dot(t_vec)
        # dval = coeffts.dot(dt_vec)
        # z = val * flag + self.walking_height + leg.z_shift
        # dz = dval * flag/self.dt

        return z, dz, flag

    def safety_check(self, ee_pos: np.ndarray) -> np.ndarray:
        '''
        Performs a safety check over the planned foot pos according to the kinematic limits of the leg.
        Calculates the corrected, safe foot pos, if required.
        Args:
            ee_pos : planned foot pos according to the cmd vel and the trajectory, with shape = (batch_size, 3)
        Ret:
            safe_ee_pos : corrected, safe foot pos if the planned foot pos was outside 90% of the workspace (extra safety), or the planned foot pos, with shape = (batch_size, 3)
        '''
        safe_ee_pos = ee_pos.copy()
        mag: np.ndarray = np.linalg.norm(safe_ee_pos,
                                         axis=1)  # Magnitude of planned foot pos vector from hip (leg origin)
        if mag.any() == 0:
            print(red("Warning: Magnitude of planned foot pos vector from hip (leg origin) is 0"))

        # Safety Calculations
        # max radius of workspace of leg, equation of sphere
        l1, l2, l3 = self.link_lengths_stoch3.tolist()
        r = np.linalg.norm(safe_ee_pos[:, 1:3], axis=1)

        mask1 = np.sum(safe_ee_pos[:, 1:3] ** 2, axis=1) < self.link_lengths_stoch3[0] ** 2
        safe_ee_pos[mask1, 1:3] = self.link_lengths_stoch3[0] * safe_ee_pos[mask1, 1:3] / mag[mask1].reshape(-1, 1)

        mask2 = (r ** 2 < (l1 - l2) ** 2) & (r ** 2 > (l1 + l2) ** 2)

        # print(r[mask2].shape)
        # print(safe_ee_pos[mask2, 0].shape)
        # print(mag[mask2].shape)
        safe_ee_pos[mask2, 0] = (0.9 * r[mask2]) * safe_ee_pos[mask2, 0] / mag[mask2]
        safe_ee_pos[mask2, 2] = (0.9 * r[mask2]) * safe_ee_pos[mask2, 2] / mag[mask2]

        return safe_ee_pos

    def initialize_leg_state(self, action: np.ndarray, prev_motor_angles: np.ndarray, dt: float) -> NamedTuple:
        '''
        Initialize all the parameters of the leg trajectories
        Args:
            action            : trajectory modulation parameters predicted by the policy, with shape = (batch_size, 18)
            prev_motor_angles : joint encoder values for the previous control step
            dt                : control period
        Ret:
            legs : namedtuple('legs', 'front_right front_left back_right back_left')
        '''

        Legs = namedtuple('legs', 'front_left front_right back_left back_right')
        legs = Legs(front_left=self.front_left, front_right=self.front_right,
                    back_left=self.back_left, back_right=self.back_right)

        self.update_leg_theta(dt)

        action = action.reshape(action.shape[0], 1, action.shape[1])
        shifts = np.concatenate([(action[:, :, 0:4]).transpose(0, 2, 1),
                                 (action[:, :, 4:8]).transpose(0, 2, 1),
                                 (action[:, :, 8:12]).transpose(0, 2, 1)], axis=2)
        self.initialize_traj_shift(shifts)
        self.initialize_prev_motor_ang(prev_motor_angles)

        return legs

    def generate_trajectory(self, action: th.Tensor,
                            prev_motor_angles: th.Tensor,
                            dt: float, foot_contacts: th.Tensor):
        '''
        Velocity based trajectory generator. The controller assumes a default trot gait.
        Note: we are using the right hand rule for the conventions of the leg which is - x->front, y->left, z->up
        TO DO:
            1. Inverse Kinematics vectorization
            2. Add joint angles, foot velocities etc. to the return
            3. Documentation
        Args:
            action : trajectory modulation parameters predicted by the policy, with shape = (batch_size, 18)
            prev_motor_angles : joint encoder values for the previous control step, with shape = (batch_size, 12)
            dt                : control period
        Ret:
            leg_motor_angles : list of motors positions for the desired action [FLH FLK FRH FRK BLH BLK BRH BRK FLA FRA BLA BRA]
        '''
        self.dt = dt
        action = action.detach().cpu().numpy()
        prev_motor_angles = prev_motor_angles.detach().cpu().numpy()
        legs = self.initialize_leg_state(action, prev_motor_angles, dt)
        aug_6D_twist: np.ndarray = action[:, 12:]

        leg: LegData
        for leg_num, leg in enumerate(legs):
            # print(f"Generating trajectory for {leg.name}", f"\nphase: {leg.phase}")
            leg.true_is_stance = foot_contacts[:, leg_num].cpu().numpy()
            leg.batch_prev_ee = leg.batch_curr_ee  # Open-loop
            xy, vxy = self.calculate_planar_traj(leg, aug_6D_twist, dt)
            z, vz, _ = self.calculate_vert_comp(leg)
            leg.stance = self.isStance(leg)
            leg.batch_curr_ee = np.concatenate([xy, z], axis=1)
            leg.batch_foot_vel = np.concatenate([vxy, vz], axis=1)

            leg.batch_curr_ee = self.safety_check(leg.batch_curr_ee)

            if leg.name == "BR" or leg.name == "br" or leg.name == "bl" or leg.name == "BL":
                branch = ">"
            else:
                branch = ">"

            valid, leg.batch_curr_motor_angles = self.stoch3_kin.inverseKinematics(leg.name, leg.batch_curr_ee.copy(),
                                                                                   branch)

            # print("foot position z",leg.batch_curr_ee)
            # if(not valid):
            #     exit()

        temp_theta = np.concatenate([
            legs.front_left.batch_curr_motor_angles,
            legs.front_right.batch_curr_motor_angles,
            legs.back_left.batch_curr_motor_angles,
            legs.back_right.batch_curr_motor_angles
        ], axis=1)
        leg_dof_tensor = th.from_numpy(temp_theta).float().to(device=self._device)

        # temp_dtheta = np.zeros((self.Nb * 3, 1), dtype=float)
        # leg_dof_tensor = th.from_numpy(
        #                     np.concatenate([temp_theta, temp_dtheta], axis=1)
        #                     ).float().to(device=self._device)

        # leg_motor_angles =
        # print(legs.front_right.frame)
        leg_foot_pos = th.cat([
            th.from_numpy(legs.front_left.batch_curr_ee + legs.front_left.frame).to(device=self._device),
            th.from_numpy(legs.front_right.batch_curr_ee + legs.front_right.frame).to(device=self._device),
            th.from_numpy(legs.back_left.batch_curr_ee + legs.back_left.frame).to(device=self._device),
            th.from_numpy(legs.back_right.batch_curr_ee + legs.back_right.frame).to(device=self._device)
        ], dim=1)

        leg_foot_vel = th.cat([
            th.from_numpy(legs.front_left.batch_foot_vel).to(device=self._device),
            th.from_numpy(legs.front_right.batch_foot_vel).to(device=self._device),
            th.from_numpy(legs.back_left.batch_foot_vel).to(device=self._device),
            th.from_numpy(legs.back_right.batch_foot_vel).to(device=self._device)
        ], dim=1)

        # print("legs front_left", legs.front_left.stance.shape)
        # print("stance bool", self.stance_bool[..., 0].shape)
        self.stance_bool[..., 0] = th.tensor(legs.front_left.stance).reshape(-1)
        self.stance_bool[..., 1] = th.tensor(legs.front_right.stance).reshape(-1)
        self.stance_bool[..., 2] = th.tensor(legs.back_left.stance).reshape(-1)
        self.stance_bool[..., 3] = th.tensor(legs.back_right.stance).reshape(-1)

        # cmd_vel = np.array([lin_vel_x, lin_vel_y, 0, 0, 0, ang_vel_z])

        # return leg_dof_tensor, leg_foot_pos, leg_foot_vel, th.from_numpy(aug_6D_twist).to(device=self._device)
        return leg_dof_tensor, leg_foot_pos, leg_foot_vel, self.stance_bool


if __name__ == '__main__':
    '''
    This script can be run independently to plot the generated trajectories in either the leg frame or the robot base frame.
    Note: To run this file independently, copy-paste the /SlopedTerrainLinearPolicy/utils folder from the
    /SlopedTerrainLinearPolicy folder to /SlopedTerrainLinearPolicy/gym_sloped_terrains/envs  
    '''
    no_of_points = 200
    batch_size = 2
    # logger = DataLog()
    trajgen = TrajectoryGenerator('cpu', batch_size=batch_size, sim='isaac')

    # a = np.array([0.6803754343094189,	-0.21123414636181398,	0.566198447517212,
    #             0.596880066952147,	0.8232947158735691,	-0.6048972614132321,
    #             -0.329554488570222,	0.536459189623808,	-0.44445057839362395,
    #             0.10793991159086098,	-0.0452058962756795,	0.257741849523849,
    #             -0.27043105441631304,	0.026801820391231003,	0, 0, 0, 0.904459450349426]).reshape(1, -1)
    # action = np.concatenate([a, a], axis=0)

    action = th.zeros((batch_size, 18), dtype=float).to(device='cpu')
    action[0, 12] = 0
    action[0, 17] = 0.5
    plotdata = []
    dt = 0.02
    prev_motor_data = th.zeros((batch_size, 4 * 3), dtype=float).to(device='cpu')
    ax = plt.axes(projection='3d')

    for i in range(no_of_points):
        _, des_foot_pos, _, _ = trajgen.generate_trajectory(action, prev_motor_data, dt)
        plotdata.append(des_foot_pos[0, :].detach().cpu().numpy())
        # logger.log_kv('x_fl', des_foot_pos[0][0])
        # logger.log_kv('y_fl', des_foot_pos[0][1])
        # logger.log_kv('z_fl', des_foot_pos[0][2])
        # logger.log_kv('x_fr', des_foot_pos[1][0])
        # logger.log_kv('y_fr', des_foot_pos[1][1])
        # logger.log_kv('z_fr', des_foot_pos[1][2])
        # logger.log_kv('x_bl', des_foot_pos[2][0])
        # logger.log_kv('y_bl', des_foot_pos[2][1])
        # logger.log_kv('z_bl', des_foot_pos[2][2])
        # logger.log_kv('x_br', des_foot_pos[3][0])
        # logger.log_kv('y_br', des_foot_pos[3][1])
        # logger.log_kv('z_br', des_foot_pos[3][2])

    x_fl = [p[0] for p in plotdata]
    y_fl = [p[1] for p in plotdata]
    z_fl = [p[2] for p in plotdata]
    x_fr = [p[3] for p in plotdata]
    y_fr = [p[4] for p in plotdata]
    z_fr = [p[5] for p in plotdata]
    x_bl = [p[6] for p in plotdata]
    y_bl = [p[7] for p in plotdata]
    z_bl = [p[8] for p in plotdata]
    x_br = [p[9] for p in plotdata]
    y_br = [p[10] for p in plotdata]
    z_br = [p[11] for p in plotdata]

    # logger.save_log('trajectory_generators', 'traj_data', 'csv')

    ax.plot3D(x_fl, y_fl, z_fl, 'red')
    ax.plot3D(x_fr, y_fr, z_fr, 'blue')
    ax.plot3D(x_bl, y_bl, z_bl, 'blue')
    ax.plot3D(x_br, y_br, z_br, 'red')

    plt.show()
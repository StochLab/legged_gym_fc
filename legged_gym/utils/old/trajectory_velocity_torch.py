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
  action[:, 0:4  ] -> x_shift: fl, fr, bl, br
  action[:, 4:8  ] -> y_shift: fl, fr, bl, br
  action[:, 8:12 ] -> z_shift: fl, fr, bl, br
  action[:, 12:15] -> cmd_vel_lin: vx, vy, vz
  action[:, 15:18] -> cmd_vel_ang: wx, wy, wz
"""

from typing import NamedTuple, Tuple
from collections import namedtuple
from legged_gym.utils.kinematics_torch import Stoch3Kinematics
# from utils.logger import DataLog
import numpy as np
import matplotlib.pyplot as plt
from fabulous.color import red

import torch

PI: float = np.pi
FL = 0
FR = 1
BL = 2
BR = 3

class LegData:
    def __init__(self, device, name: str, ID: int, frame: torch.Tensor, Nb: int = 1):
        self.name = name
        self.ID = ID
        self.frame = frame.float()
        self.Nb = Nb
        self.theta: float = 0.0
        self.batch_prev_motor_angles: torch.Tensor = torch.zeros((Nb, 3), dtype=float).to(device)
        self.batch_curr_motor_angles: torch.Tensor = torch.zeros((Nb, 3), dtype=float).to(device)
        self.batch_stance_bool: torch.Tensor = torch.zeros((Nb, 1), dtype=bool).to(device)
        self.batch_prev_ee: torch.Tensor = torch.zeros((Nb, 3), dtype=float).to(device)
        self.batch_curr_ee: torch.Tensor = torch.zeros((Nb, 3), dtype=float).to(device)
        self.batch_shifts: torch.Tensor = torch.zeros((Nb, 3), dtype=float).to(device)
        self.batch_foot_vel: torch.Tensor = torch.zeros((Nb, 3), dtype=float).to(device)

class GaitData:
    """
        frequency: torch.Tensor = 2.5,
        max_linear_xvel: torch.Tensor = 2.5/5, max_linear_yvel: torch.Tensor = 2.5/10,
        max_angular_vel: torch.Tensor = 2.5, swing_height: torch.Tensor = 0.060,
        torso_height: torch.Tensor = -0.25, stance_start: torch.Tensor = [0, PI, PI, 0]
        stance_duration: torch.Tensor = [PI, PI, PI, PI]
    """

    def __init__(self, frequency: float = 2.5, swing_height: float = 0.08,
                 torso_height: float = -0.45, stance_start: torch.Tensor = [0, PI, PI, 0],
                 stance_duration: torch.Tensor = [PI, PI, PI, PI]):

        """Creates trot as default"""
        self.frequency = frequency
        self.max_linear_xvel = 1.0  # frequency/5
        self.max_linear_yvel = 0.80  # frequency/5
        self.max_angular_vel = 1.25  # frequency/2
        self.swing_height = swing_height
        self.torso_height = torso_height
        self.stance_start = stance_start
        self.stance_duration = stance_duration
        self.omega = self.frequency * 2 * PI

    def set_new_gait_constrained(self, frequency: float, swing_height: float,
                                 torso_height: float, stance_start: torch.Tensor,
                                 stance_duration: torch.Tensor):
        """
        Function to set new gait parameters in a constrained fashion.
        Args:
            frequency : frequency of gait
            swing_height : height of swing foot
            torso_height : height of torso (-negative)
        """
        self.frequency = frequency
        self.omega = self.frequency * 2 * PI
        self.max_linear_xvel = 0.025/frequency
        self.max_linear_yvel = 0.025/frequency
        self.max_angular_vel = 1 / frequency
        self.swing_height = swing_height
        self.torso_height = torso_height
        self.stance_start = stance_start
        self.stance_duration = stance_duration

    def set_new_gait_parameter(self, frequency, swing_height):
        """
        Function to set new gait parameters in a constrained fashion.
        Args:
            frequency : frequency of gait
            swing_height : height of swing foot
        """
        self.frequency = frequency
        self.omega = self.frequency * 2 * PI
        self.max_linear_xvel = 0.025/frequency
        self.max_linear_yvel = 0.025/frequency
        self.max_angular_vel = 1 / frequency
        self.swing_height = swing_height

        self.max_6D_twist = torch.cat([self.max_linear_xvel,
                                self.max_linear_yvel, 
                                self.max_linear_yvel,  # z-vel
                                self.max_angular_vel/5, 
                                self.max_angular_vel/5, 
                                self.max_angular_vel], dim=1)

class TrajectoryGenerator():
    """
    A class that generates swing and stance trajectories for
    walking based on input velocity.
    Note: An assumption is made that the trajectory cycle
          theta is the same for all the batches
    """
    def __init__(self, device, batch_size=1, sim="isaac", gait_type='trot'):
        self._device = device
        self.Nb: int = batch_size
        self.theta = torch.zeros((batch_size, 1),
                                 dtype=torch.float32).to(self._device)  # self.theta = 0.0

        self.stoch3_kin = Stoch3Kinematics()
        self.robot_width = self.stoch3_kin.torso_dims[1]
        self.robot_length = self.stoch3_kin.torso_dims[0]
        self.link_lengths_stoch3 = torch.tensor(self.stoch3_kin.link_lengths).to(self._device)

        self.front_left = LegData(device, 'FL', FL,
                                  self.stoch3_kin.leg_frames[0, ...].reshape((1, 3)), batch_size)
        self.front_right = LegData(device, 'FR', FR,
                                   self.stoch3_kin.leg_frames[1, ...].reshape((1, 3)), batch_size)
        self.back_left = LegData(device, 'BL', BL,
                                 self.stoch3_kin.leg_frames[2, ...].reshape((1, 3)), batch_size)
        self.back_right = LegData(device, 'BR', BR,
                                  self.stoch3_kin.leg_frames[3, ...].reshape((1, 3)), batch_size)

        # warning:: changing swing height from 0.15
        if gait_type == 'trot':
            self.gait = GaitData(frequency=2.5, swing_height=0.15,
                                 torso_height=-0.45,
                                 stance_start=[PI, 0, 0, PI],
                                 stance_duration=[PI, PI, PI, PI])
        elif gait_type == 'crawl':
            self.gait = GaitData(frequency=0.5, swing_height=0.15,
                                 torso_height=-0.45,
                                 stance_start=[5*PI/3, 2*PI/3, 4*PI/3, 1*PI/3],
                                 stance_duration=[5.1*PI/3, 5.1*PI/3, 5.1*PI/3, 5.1*PI/3])
        else:
            print(red("Unknown Gait Type, selecting default trot gait"))
            self.gait = GaitData(frequency=2.5, swing_height=0.15,
                                 torso_height=-0.45,
                                 stance_start=[0, PI, PI, 0],
                                 stance_duration=[PI, PI, PI, PI])

        self.max_6D_twist = torch.tensor([self.gait.max_linear_xvel,
                                          self.gait.max_linear_yvel,
                                          self.gait.max_linear_yvel,  # z-vel
                                          self.gait.max_angular_vel/5,
                                          self.gait.max_angular_vel/5,
                                          self.gait.max_angular_vel]).to(self._device)
        self.sim = sim

        self.stance_bool = torch.cat([self.front_left.batch_stance_bool,
                                      self.front_right.batch_stance_bool,
                                      self.back_left.batch_stance_bool,
                                      self.back_right.batch_stance_bool], dim=1).bool().to(device=self._device)

    def constrain_theta(self, theta: float) -> float:
        """
        A function to constrain theta between [0, 2*Pi]
        
        Args:
            theta : trajectory cycling parameter
        """
        theta = torch.fmod(theta, 2 * PI)
        # if theta < 0, then theta = theta + 2*PI
        theta = theta + (theta < 0) * (2 * PI * torch.ones(theta.shape,
                                                           dtype=torch.float32).to(self._device))
        return theta

    def update_leg_theta(self, dt: float):
        """
        Function to calculate the leg cycles of the trajectory in batches,
        depending on the gait.
        Args:
            dt : time step
        """
        # print("theta",self.theta.shape)
        # print("gait omega",self.gait.omega.device)
        self.theta: torch.Tensor = self.constrain_theta(self.theta + self.gait.omega * dt) 

    def reset_theta(self, idx):
        """
        Function to reset the main cycle of the trajectory.
        """
        self.theta[idx, 0] = 0.0

    def initialize_traj_shift(self, shifts: torch.Tensor):
        """
        Initialize desired <x,y,z> offsets of trajectory for each leg
        Args:
            shifts : Translational shifts as an Nd-array of size = (batch_size, num_legs, 3)
        """
        self.front_left.batch_shifts = shifts[:, FL, :].reshape((self.Nb, 3))
        self.front_right.batch_shifts = shifts[:, FR, :].reshape((self.Nb, 3))
        self.back_left.batch_shifts = shifts[:, BL, :].reshape((self.Nb, 3))
        self.back_right.batch_shifts = shifts[:, BR, :].reshape((self.Nb, 3))
        
    def initialize_prev_motor_ang(self, prev_motor_angles: torch.Tensor):
        """
        Initialize motor angles of previous time-step for each leg
        Args:
            prev_motor_angles : Previous time step encoder recorded joint angles as an
                                Nd-array of size (batch_size, 3*num_legs)
        """
        self.front_left.batch_prev_motor_angles = prev_motor_angles[:, 0:3].reshape((self.Nb, 3))
        self.front_right.batch_prev_motor_angles = prev_motor_angles[:, 3:6].reshape((self.Nb, 3))
        self.back_left.batch_prev_motor_angles = prev_motor_angles[:, 6:9].reshape((self.Nb, 3))
        self.back_right.batch_prev_motor_angles = prev_motor_angles[:, 9:12].reshape((self.Nb, 3))

    def foot_step_planner(self, leg: LegData, v_leg: torch.Tensor) -> torch.Tensor:
        """
        Calculates the  absolute coordinate (wrt hip frame) where the foot should land at the beginning of the stance phase of the trajectory based on the 
        commanded velocities (either from joystick or augmented by the policy).
        Args:
            leg   : the leg for which the trajectory has to be calculated
            v_leg : the velocity vector for the leg (summation of linear and angular velocity components), with shape = (batch_size, 3)
        Ret:
            s : absolute coordinate of the footstep, with shape = (batch_size, 3)
        """
        stance_time: float = (self.gait.stance_duration[leg.ID] / (2 * PI)) * (1.0 / self.gait.frequency)

        # print("v leg", v_leg.shape)
        # print("stance time",stance_time.shape)
        # print("leg batch shift", leg.batch_shifts.shape)

        s: torch.Tensor = v_leg * stance_time/2 + leg.batch_shifts

        return s

    def getLegPhase(self, leg: LegData) -> float:
        leg_phase = self.theta - self.gait.stance_start[leg.ID]
        # if (leg_phase < 0):
        #     leg_phase += 2 * PI
        leg_phase = leg_phase + (leg_phase < 0) * (2 * PI * torch.ones(leg_phase.shape).to(self._device))
        return leg_phase.reshape(-1, 1)

    def isStance(self, leg: LegData):
        leg_phase = self.getLegPhase(leg)
        # Having <= gives a single time-step when all the legs are in contact
        # if (leg_phase <= self.gait.stance_duration[leg.ID]):
        #     return True
        # else:
        #     return False
        return (leg_phase <= self.gait.stance_duration[leg.ID]).reshape(-1, 1)

    def calculate_planar_traj(self, leg: LegData, aug_6D_twist: torch.Tensor,
                              dt: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the x and y component of the trajectory based on the commanded velocities
        (either from joystick or augmented by the policy).
        Args:
            leg          : the leg for which the trajectory has to be calculated
            aug_6D_twist : augmented body twist commanded by the policies,
                           with shape = (batch_size, 6)
            dt           : control period
        Ret:
            r : calculated x and y coordinates of the trajectory in batches.
            v : calculated vx and vy linear of the trajectory in batches.
        """
        
        # Since we will consider only the 2D motion of the robot, we will ignore the linear z component of the twist
        cmd_lvel = torch.cat([aug_6D_twist[:, 0:2], torch.zeros((self.Nb, 1)).to(self._device)], dim=1)
        # Since we will consider only the 2D motion of the robot, we will consider the angular z component of the twist
        cmd_avel = torch.cat([torch.zeros((self.Nb, 2)).to(self._device), aug_6D_twist[:, 5].reshape(-1, 1)], dim=1)

        # Only consider linear velocity of the foot
        v_leg = torch.zeros((self.Nb, 3), dtype=float).to(self._device)
        next_step = torch.zeros((self.Nb, 3), dtype=float).to(self._device)
        swing_vec = torch.tensor((self.Nb, 3), dtype=float).to(self._device)

        prev_foot_pos = torch.cat([leg.batch_prev_ee[:,0:2],
                                   torch.zeros((self.Nb, 1), dtype=float).to(self._device)],
                                  dim=1).reshape((self.Nb, 3))
        
        # print("leg.frame",leg.frame)
        # print("prev_foot_pos",type(prev_foot_pos))
        # prev_r = prev_foot_pos

        prev_r = prev_foot_pos.float() + leg.frame
        v_lcomp = cmd_lvel

        # print("prev foot pos",prev_foot_pos.dtype)
        # print("cmd avel",cmd_avel.dtype)
        # print("prev r", prev_r.dtype)

        v_acomp = torch.cross(cmd_avel, prev_r)
        # if(leg.name == 'FL'):
        #   print("leg position",prev_r)
        v_leg = v_lcomp + v_acomp

        isStance_bool = self.isStance(leg)

        # if self.isStance(leg):
        #     flag = -1  # during stance_phase of walking, leg moves backwards to push body forward
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

        flag = -1 * isStance_bool + (isStance_bool is False)

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
        dr_2 = (time_left > 1e-4) * swing_vec / time_left * dt * flag * (isStance_bool is False)
        r_2 = prev_r + dr_2 - leg.frame

        r = r_1 * isStance_bool + r_2 * (isStance_bool is False)
        
        # print("time left",time_left,"dr2",dr_2,"theta",self.theta)
        # print("next_step",next_step,"prev_r",prev_r)

        return r[:, 0:2], flag*v_leg[:, 0:2]

    def cspline_coeff(self, z0: torch.Tensor, z1: torch.Tensor,
                      d0: torch.Tensor, d1: torch.Tensor, t: float):
        """
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
        """
        coefft = torch.zeros((self.Nb, 4), dtype=float).to(self._device)
        coefft[:, 0] = z0
        coefft[:, 1] = d0
        w0 = z1 - z0 - d0*t
        w1 = d1 - d0
        coefft[:, 2] = -1*(-3*t**2*w0 + t**3*w1)/t**4
        coefft[:, 3] = -1*(2*t*w0 - t**2*w1)/t**4

        return coefft

    def calculate_vert_comp(self, leg: LegData) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Calculates the z component of the trajectory. The function for the z component can be changed here.
        The z component calculation is kept independent as it is not affected by the velocity calculations. 
        Various functions can be used to smoothen out the foot impacts while walking.
        Args:
            leg : the leg for which the trajectory has to be calculated
        Ret:
            z    : calculated z component of the trajectory in batches
            dz   : linear velocity component of the trajectory in batches
            flag : flag for the virtual (assumed) foot contact
        """

        # theta taken from +x, CW, flip this sigh if the trajectory is mirrored
        # if not self.isStance(leg):
        #     flag = 1
        # else:
        #     flag = 0

        flag = (self.isStance(leg) is False)

        swing_phase = self.getLegPhase(leg) - self.gait.stance_duration[leg.ID]
        swing_duration = 2 * PI - self.gait.stance_duration[leg.ID]
        theta_leg = PI * swing_phase / swing_duration
        dtheta_leg = PI * self.gait.omega / swing_duration
        
        # Sine function
        z: torch.Tensor = self.gait.swing_height * torch.sin(theta_leg) * \
            flag * torch.ones((self.Nb, 1)).to(self._device) + self.gait.torso_height

        # if(z[0][0]>-0.45):
        #    print("z",z,"leg",leg.name,"theta::",theta_leg)

        # dz/dt = dz/dtheta * dtheta/dt
        dz: torch.Tensor = self.gait.swing_height * torch.cos(theta_leg) * \
            flag * torch.ones((self.Nb, 1)).to(self._device) * dtheta_leg
         
        """ CONVERSION TO BATCH FORM LEFT """
        '''
        Cubic Spline
        This cubic spline is defined by 5 control points (n=4). Each control point is (theta, z) ie (theta_0, z_0) to (theta_n, z_n) n+1 control points
        The assumed cubic spline is of the type:
        z = coefft_n[3]*(t-t_n)**3 + coefft_n[2]*(t-t_n)**2 + coefft_n[1]*(t-t_n)**1 + coefft_n[0]*(t-t_n)**0 {0<=n<=3}
        where, n denotes each section of the cubic spline, governed by the nth-index control point 
        '''
        # theta = [0, PI/4, PI/2, 3*PI/4, PI]
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

    def safety_check(self, ee_pos: torch.Tensor) -> torch.Tensor:
        """
        Performs a safety check over the planned foot pos according to the kinematic
        limits of the leg.  Calculates the corrected, safe foot pos, if required.
        Args:
            ee_pos : planned foot pos according to the cmd vel and the trajectory,
            with shape = (batch_size, 3)
        Ret:
            safe_ee_pos : corrected, safe foot pos if the planned foot pos was outside 90% of
            the workspace (extra safety), or the planned foot pos, with shape = (batch_size, 3)
        """
        safe_ee_pos = ee_pos.clone()
        # Magnitude of planned foot pos vector from hip (leg origin)
        mag:torch.Tensor = torch.norm(safe_ee_pos, dim=1)
        if mag.any() == 0:
            print(red("Warning: Magnitude of planned foot pos vector from hip (leg origin) is 0"))
    
        # Safety Calculations
        # max radius of workspace of leg, equation of sphere
        l1, l2, l3 = self.link_lengths_stoch3.tolist()
        r = torch.norm(safe_ee_pos[:,1:3], dim=1)
        
        mask1 = torch.sum(safe_ee_pos[:, 1:3]**2, dim=1) < self.link_lengths_stoch3[0]**2
        safe_ee_pos[mask1, 1:3] = self.link_lengths_stoch3[0] * safe_ee_pos[mask1, 1:3] / mag[mask1].reshape(-1, 1)

        mask2 = (r**2 < (l1-l2)**2) & (r**2 > (l1+l2)**2)

        # print(r[mask2].shape)
        # print(safe_ee_pos[mask2, 0].shape)
        # print(mag[mask2].shape)

        safe_ee_pos[mask2, 0] = (0.9*r[mask2]) * safe_ee_pos[mask2, 0] / mag[mask2]
        safe_ee_pos[mask2, 2] = (0.9*r[mask2]) * safe_ee_pos[mask2, 2] / mag[mask2]

        return safe_ee_pos

    def initialize_leg_state(self, action: torch.Tensor,
                             prev_motor_angles: torch.Tensor, dt: float) -> NamedTuple:
        """
        Initialize all the parameters of the leg trajectories
        Args:
            action            : trajectory modulation parameters predicted by the policy,
                                with shape = (batch_size, 18)
            prev_motor_angles : joint encoder values for the previous control step
            dt                : control period 
        Ret:
            legs : namedtuple('legs', 'front_right front_left back_right back_left')
        """
        Legs = namedtuple('legs', 'front_left front_right back_left back_right')
        legs = Legs(front_left=self.front_left, front_right=self.front_right,
                    back_left=self.back_left,  back_right=self.back_right)

        self.update_leg_theta(dt)

        action = action.reshape(action.shape[0], 1, action.shape[1])
        shifts = torch.cat([(action[:, :, 0:4]).transpose(2, 1),
                            (action[:, :, 4:8]).transpose(2, 1),
                            (action[:, :, 8:12]).transpose(2, 1)], dim=2)
        self.initialize_traj_shift(shifts)
        self.initialize_prev_motor_ang(prev_motor_angles)

        return legs

    def generate_trajectory(self, action: torch.Tensor,
                            prev_motor_angles: torch.Tensor, dt: float):
        """
        Velocity based trajectory generator. The controller assumes a default trot gait. 
        Note: we are using the right hand rule for the conventions of the leg
              which is - x->front, y->left, z->up
        TO DO:
            1. Inverse Kinematics vectorization
            2. Add joint angles, foot velocities etc. to the return
            3. Documentation
        Args:
            action : trajectory modulation parameters predicted by the policy, with shape = (batch_size, 18)
            prev_motor_angles : joint encoder values for the previous control step, with shape = (batch_size, 12)
            dt : control period
        Ret:
            leg_motor_angles : list of motors positions for the desired action
                               [FLH, FLK, FRH, FRK, BLH, BLK, BRH, BRK, FLA, FRA, BLA, BRA]
        """
        self.dt = dt
        action = action
        prev_motor_angles = prev_motor_angles
        legs = self.initialize_leg_state(action, prev_motor_angles, dt)
        aug_6D_twist: torch.Tensor = action[:, 12:] # * self.max_6D_twist

        leg: LegData
        for leg in legs:
            # print(f"Generating trajectory for {leg.name}", f"\n phase: {leg.phase}")
            leg.batch_prev_ee = leg.batch_curr_ee  # Open-loop
            xy, vxy = self.calculate_planar_traj(leg, aug_6D_twist, dt)
            z, vz, _ = self.calculate_vert_comp(leg)
            leg.stance = self.isStance(leg)
            leg.batch_curr_ee = torch.cat([xy, z], dim=1)
            leg.batch_foot_vel = torch.cat([vxy, vz], dim=1)
            leg.batch_curr_ee = self.safety_check(leg.batch_curr_ee)

            if leg.name == "BR" or leg.name == "br" or leg.name == "bl" or leg.name == "BL":
                branch = ">"
            else:
                branch = ">"

            valid, leg.batch_curr_motor_angles = self.stoch3_kin.inverseKinematics(leg.name, leg.batch_curr_ee.clone(), branch)

            # print("foot position z",leg.batch_curr_ee)
            # if(not valid):
            #     exit()

        temp_theta = torch.cat([legs.front_left.batch_curr_motor_angles,
                                legs.front_right.batch_curr_motor_angles,
                                legs.back_left.batch_curr_motor_angles,
                                legs.back_right.batch_curr_motor_angles], dim=1)
        leg_dof_tensor = temp_theta.float().to(device=self._device)

        # temp_dtheta = torch.zeros((self.Nb * 3, 1), dtype=float)
        # leg_dof_tensor = torch.from_numpy(
        #                     np.concatenate([temp_theta, temp_dtheta], axis=1)
        #                     ).float().to(device=self._device)
        # leg_motor_angles = 
        # print(legs.front_right.frame)

        leg_foot_pos = torch.cat([
            (legs.front_left.batch_curr_ee + legs.front_left.frame).to(device=self._device),
            (legs.front_right.batch_curr_ee + legs.front_right.frame).to(device=self._device),
            (legs.back_left.batch_curr_ee + legs.back_left.frame).to(device=self._device),
            (legs.back_right.batch_curr_ee + legs.back_right.frame).to(device=self._device)
        ], dim=1)

        leg_foot_vel = torch.cat([
            (legs.front_left.batch_foot_vel).to(device=self._device),
            (legs.front_right.batch_foot_vel).to(device=self._device),
            (legs.back_left.batch_foot_vel).to(device=self._device),
            (legs.back_right.batch_foot_vel).to(device=self._device)
        ], dim=1)

        # print("legs front_left", legs.front_left.stance.shape)
        # print("stance bool", self.stance_bool[..., 0].shape)

        self.stance_bool[..., 0] = legs.front_left.stance.reshape(-1)
        self.stance_bool[..., 1] = legs.front_right.stance.reshape(-1) 
        self.stance_bool[..., 2] = legs.back_left.stance.reshape(-1)
        self.stance_bool[..., 3] = legs.back_right.stance.reshape(-1)

        # cmd_vel = np.array([lin_vel_x, lin_vel_y, 0, 0, 0, ang_vel_z])

        return leg_dof_tensor, leg_foot_pos, leg_foot_vel, aug_6D_twist
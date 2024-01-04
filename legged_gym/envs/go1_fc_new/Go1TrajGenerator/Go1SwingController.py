from legged_gym.go1_scripts.Go1TrajGenerator.Go1Gait import OffsetDurationGait
from legged_gym.go1_scripts.Go1TrajGenerator.Go1FootSwingTrajectory import FootSwingTrajectory
from Quadruped import Quadruped
from legged_gym.go1_scripts.go1_utils import getSideSign
from legged_gym.go1_scripts.Go1TrajGenerator.StateEstimator import StateEstimator
from legged_gym.go1_scripts.Go1TrajGenerator.LegController import LegController
from math_utils.orientation_tools import coordinateRotation, CoordinateAxis
import numpy as np

DTYPE = np.float32
CASTING = "same_kind"
cmpc_bonus_swing = 0.0
flat_ground = True

class SwingLegController:

    def __init__(self, gaitNumber, _dt):
        # pick gait

        self.gait = OffsetDurationGait(10,
                                      np.array([0, 5, 5, 0], dtype=DTYPE),
                                      np.array([5, 5, 5, 5], dtype=DTYPE), "Trotting")
        if gaitNumber == 1:
            self.gait = OffsetDurationGait(10,
                                      np.array([5, 5, 0, 0], dtype=DTYPE),
                                      np.array([4, 4, 4, 4], dtype=DTYPE), "Bounding")
        elif gaitNumber == 2:
            self.gait = OffsetDurationGait(10,
                                      np.array([0, 0, 0, 0], dtype=DTYPE),
                                      np.array([4, 4, 4, 4], dtype=DTYPE), "Pronking")
        elif gaitNumber == 3:
            self.gait = OffsetDurationGait(10,
                                    np.array([5, 0, 5, 0], dtype=DTYPE),
                                    np.array([5, 5, 5, 5], dtype=DTYPE), "Pacing")
        elif gaitNumber == 5:
            self.gait = OffsetDurationGait(10,
                                       np.array([0, 2, 7, 9], dtype=DTYPE),
                                       np.array([4, 4, 4, 4], dtype=DTYPE), "Galloping")
        elif gaitNumber == 6:
            self.gait = OffsetDurationGait(10,
                                     np.array([0, 3, 5, 8], dtype=DTYPE),
                                     np.array([5, 5, 5, 5], dtype=DTYPE), "Walking")
        elif gaitNumber == 7:
            self.gait = OffsetDurationGait(10,
                                         np.array([0, 5, 5, 0], dtype=DTYPE),
                                         np.array([4, 4, 4, 4], dtype=DTYPE), "Trot Running")

        self.dt = _dt
        self.iterationsBetweenMPC = 2
        self.dtMPC = self.dt * self.iterationsBetweenMPC
        self.default_iterations_between_mpc = self.iterationsBetweenMPC
        self.firstSwing = [True for _ in range(4)]
        self.firstRun = True
        self.iterationCounter = 0
        self.pFoot = np.zeros((4, 3, 1), dtype=DTYPE)
        self.foot_positions = np.zeros((4, 3, 1), dtype=DTYPE)

        self._quadruped = Quadruped()
        self._legController = LegController(self._quadruped)
        self._stateEstimator = StateEstimator(self._quadruped)

        self.current_gait = 0
        self._x_vel_des = 0.0
        self._y_vel_des = 0.0
        self._yaw_turn_rate = 0.0

        self.contactStates = self.gait.getContactState()
        self.swingStates = self.gait.getSwingState()
        self.mpcTable = self.gait.getMpcTable()

        self.footSwingTrajectories = [FootSwingTrajectory() for _ in range(4)]
        self.swingTimes = np.zeros((4, 1), dtype=DTYPE)
        self.swingTimeRemaining = [0.0 for _ in range(4)]

        self.Kp = np.array([700, 0, 0, 0, 700, 0, 0, 0, 150], dtype=DTYPE).reshape((3,3))
        self.Kd = np.array([7, 0, 0, 0, 7, 0, 0, 0, 7], dtype=DTYPE).reshape((3,3))
        self.Kp_stance = np.zeros_like(self.Kp)  # self.Kp
        self.Kd_stance = self.Kd

    def recomputerTiming(self, iterations_per_mpc:int):
        self.iterationsBetweenMPC = iterations_per_mpc
        self.dtMPC = self.dt*iterations_per_mpc

    def __SetupCommand(self, commands: np.array):
        self._x_vel_des = commands[0]
        self._y_vel_des = commands[1]
        self._yaw_turn_rate = commands[2]

    def run_controller(self, dof_states, body_states, commands):
        print('------------Normal-------------')
        self.__SetupCommand(commands)
        self._legController.updateData(dof_states)
        self._stateEstimator.update(body_states)

        seResult = self._stateEstimator.getResult()
        # print('seResult pos: ', seResult.position.flatten())

        self.gait.setIterations(self.iterationsBetweenMPC, self.iterationCounter)
        self.recomputerTiming(self.default_iterations_between_mpc)

        for i in range(4):
            self.foot_positions[i] = self._quadruped.getHipLocation(i) + self._legController.datas[i].p
            self.pFoot[i] = self.foot_positions[i] + seResult.position
            # print('leg pos: ', self.pFoot[i].flatten())

        # print('pFoot: ', self.pFoot, self.pFoot.shape)
        # * first time initialization
        if self.firstRun:
            self.firstRun = False
            self._stateEstimator._init_contact_history(self.foot_positions)
            for i in range(4):
                self.footSwingTrajectories[i].setHeight(0.15)
                self.footSwingTrajectories[i].setInitialPosition(self.pFoot[i])
                self.footSwingTrajectories[i].setFinalPosition(self.pFoot[i])

        if flat_ground:
            self._stateEstimator._update_com_position_ground_frame(self.foot_positions)
        else:
            self._stateEstimator._compute_ground_normal_and_com_position(self.foot_positions)

        # * foot placement
        for leg in range(4):
            self.swingTimes[leg] = self.gait.getCurrentSwingTime(self.dtMPC, leg)

        v_des_robot = np.array([self._x_vel_des, self._y_vel_des, 0],
                               dtype=DTYPE).reshape((3, 1))
        for i in range(4):
            if self.firstSwing[i]:
                self.swingTimeRemaining[i] = self.swingTimes[i].item()
            else:
                self.swingTimeRemaining[i] -= self.dt

            self.footSwingTrajectories[i].setHeight(0.15)

            offset = np.array([0, getSideSign(i) * self._quadruped._abadLinkLength, 0], dtype=DTYPE).reshape((3, 1))
            pRobotFrame = self._quadruped.getHipLocation(i) + offset
            # pRobotFrame[1] += interleave_y[i] * v_abs * interleave_gain
            stance_time = self.gait.getCurrentStanceTime(self.dtMPC, i)
            pYawCorrected = coordinateRotation(CoordinateAxis.Z,
                                               -self._yaw_turn_rate * stance_time / 2) @ pRobotFrame

            Pf = seResult.position + (pYawCorrected + v_des_robot * self.swingTimeRemaining[i])

            p_rel_max = 0.3
            pfx_rel = seResult.vBody[0] * (0.5 + cmpc_bonus_swing) * stance_time + \
                      0.03 * (seResult.vBody[0] - v_des_robot[0]) + \
                      (0.5 * seResult.position[2] / 9.81) * (seResult.vBody[1] * self._yaw_turn_rate)

            pfy_rel = seResult.vBody[1] * 0.5 * stance_time * self.dtMPC + \
                      0.03 * (seResult.vBody[1] - v_des_robot[1]) + \
                      (0.5 * seResult.position[2] / 9.81) * (-seResult.vBody[0] * self._yaw_turn_rate)

            pfx_rel = min(max(pfx_rel, -p_rel_max), p_rel_max)
            pfy_rel = min(max(pfy_rel, -p_rel_max), p_rel_max)
            Pf[0] += pfx_rel
            Pf[1] += pfy_rel
            Pf[2] = -0.003

            # print('leg: ', Pf.flatten())

            self.footSwingTrajectories[i].setFinalPosition(Pf)

        # calc gait
        self.iterationCounter += 1

        # gait
        self.contactStates = self.gait.getContactState()
        self.swingStates = self.gait.getSwingState()
        self.mpcTable = self.gait.getMpcTable()

        # print('contact states: ', self.contactStates.flatten())
        # print('swing states: ', self.swingStates.flatten())
        # print('pFoot: ', self.pFoot, self.pFoot.shape)

        se_contactState = np.array([0, 0, 0, 0], dtype=DTYPE).reshape((4, 1))

        for foot in range(4):
            contactState = self.contactStates[foot]
            swingState = self.swingStates[foot]
            if swingState > 0:  # * foot is in swing
                # print('entered first if', foot)
                if self.firstSwing[foot]:
                    # print('entered second if', foot)
                    # print('pFoot: ', self.pFoot[foot].flatten())
                    self.firstSwing[foot] = False
                    self.footSwingTrajectories[foot].setInitialPosition(self.pFoot[foot])

                self.footSwingTrajectories[foot].computeSwingTrajectoryBezier(swingState, self.swingTimes[foot].item())
                pDesFoot = self.footSwingTrajectories[foot].getPosition()
                vDesFoot = self.footSwingTrajectories[foot].getVelocity()

                pDesLeg = (pDesFoot - seResult.position) \
                          - self._quadruped.getHipLocation(foot)
                vDesLeg = (vDesFoot - seResult.vBody)

                np.copyto(self._legController.commands[foot].pDes, pDesLeg, casting=CASTING)
                np.copyto(self._legController.commands[foot].vDes, vDesLeg, casting=CASTING)
                np.copyto(self._legController.commands[foot].kpCartesian, self.Kp, casting=CASTING)
                np.copyto(self._legController.commands[foot].kdCartesian, self.Kd, casting=CASTING)

            else:  # * foot is in stance
                self.firstSwing[foot] = True
                pDesFoot = self.footSwingTrajectories[foot].getPosition()
                vDesFoot = self.footSwingTrajectories[foot].getVelocity()

                pDesLeg = (pDesFoot - seResult.position) \
                          - self._quadruped.getHipLocation(foot)
                vDesLeg = (vDesFoot - seResult.vBody)

                np.copyto(self._legController.commands[foot].pDes, pDesLeg, casting=CASTING)
                np.copyto(self._legController.commands[foot].vDes, vDesLeg, casting=CASTING)
                np.copyto(self._legController.commands[foot].kpCartesian, self.Kp_stance, casting=CASTING)
                np.copyto(self._legController.commands[foot].kdCartesian, self.Kd_stance, casting=CASTING)
                np.copyto(self._legController.commands[foot].kdJoint, np.identity(3, dtype=DTYPE) * 0.2,
                          casting=CASTING)

                se_contactState[foot] = contactState

        self._stateEstimator.setContactPhase(se_contactState)

        # prepare the data to return
        contacts = np.ceil(self._stateEstimator.getContactPhase())
        jacobian = [self._legController.datas[i].J for i in range(4)]
        foot_pos = np.vstack([self._legController.datas[i].p.flatten() for i in range(4)])
        foot_vel = np.vstack([self._legController.datas[i].v.flatten() for i in range(4)])
        des_foot_pos = np.vstack([self._legController.commands[i].pDes.flatten() for i in range(4)])
        des_foot_vel = np.vstack([self._legController.commands[i].vDes.flatten() for i in range(4)])

        mpc_inputs = dict()
        mpc_inputs['mpcTable'] = self.mpcTable
        mpc_inputs['foot_positions'] = self.foot_positions
        mpc_inputs['seResult'] = seResult

        swing_outputs = dict()
        swing_outputs['contacts'] = contacts
        swing_outputs['jacobian'] = jacobian
        swing_outputs['foot_pos'] = foot_pos
        swing_outputs['foot_vel'] = foot_vel
        swing_outputs['des_foot_pos'] = des_foot_pos
        swing_outputs['des_foot_vel'] = des_foot_vel


        return mpc_inputs, swing_outputs









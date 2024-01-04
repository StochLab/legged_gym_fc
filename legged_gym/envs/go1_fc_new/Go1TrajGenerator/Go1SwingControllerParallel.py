import numpy as np
import torch

from legged_gym.go1_scripts.Go1TrajGenerator.Go1Gait import OffsetDurationGait
from legged_gym.go1_scripts.Go1TrajGenerator.Go1FootSwingTrajectoryParallel import FootSwingTrajectory
from Quadruped import Quadruped
from legged_gym.go1_scripts.go1_utils import getSideSign
from legged_gym.go1_scripts.Go1TrajGenerator.StateEstimatorParallel import StateEstimator
from legged_gym.go1_scripts.Go1TrajGenerator.LegControllerParallel import LegController
from legged_gym.go1_scripts.math_utils.orientation_tools import coordinateRotation, CoordinateAxis

DTYPE = np.float32
CASTING = "same_kind"
cmpc_bonus_swing = 0.0
flat_ground = True

class SwingLegController:
    def __init__(self, gaitNumber, _dt, batch_size):
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

        self.batch_size = batch_size
        self.dt = _dt
        self.iterationsBetweenMPC = 2
        self.dtMPC = self.dt * self.iterationsBetweenMPC
        self.default_iterations_between_mpc = self.iterationsBetweenMPC

        # self.firstSwing = np.vstack([True for _ in range(4)] * batch_size)
        self.firstSwing = [True for _ in range(4)]
        self.firstRun = True
        self.iterationCounter = 0

        self.pFoot = np.zeros((batch_size, 4, 3), dtype=DTYPE)
        self.foot_positions = np.zeros((batch_size, 4, 3), dtype=DTYPE)

        self._quadruped = Quadruped()
        self._legController = LegController(self._quadruped, self.batch_size)
        self._stateEstimator = StateEstimator(self._quadruped, self.batch_size)

        self._x_vel_des = np.array([0.0] * self.batch_size)
        self._y_vel_des = np.array([0.0] * self.batch_size)
        self._yaw_turn_rate = np.array([0.0] * self.batch_size)

        self.contactStates = self.gait.getContactState()
        self.swingStates = self.gait.getSwingState()
        self.mpcTable = self.gait.getMpcTable()

        self.footSwingTrajectories = [FootSwingTrajectory(self.batch_size) for _ in range(4)]
        self.swingTimes = np.zeros((4, 1), dtype=DTYPE)
        self.swingTimeRemaining = [0.0 for _ in range(4)]

        # self.swingTimes = np.zeros((batch_size, 4, 1), dtype=DTYPE)
        # self.swingTimeRemaining = [[0.0] * batch_size for _ in range(4)]

    def recomputerTiming(self, iterations_per_mpc: int):
        self.iterationsBetweenMPC = iterations_per_mpc
        self.dtMPC = self.dt * iterations_per_mpc

    def __SetupCommand(self, commands: np.array):
        self._x_vel_des = commands[:, 0]
        self._y_vel_des = commands[:, 1]
        self._yaw_turn_rate = commands[:, 2]

    def run_controller(self, dof_pos, dof_vel, body_states, commands):
        self.__SetupCommand(commands)
        self._legController.updateData(dof_pos, dof_vel)
        self._stateEstimator.update(body_states)

        seResult = self._stateEstimator.getResult()

        self.gait.setIterations(self.iterationsBetweenMPC, self.iterationCounter)
        self.recomputerTiming(self.default_iterations_between_mpc)

        for i in range(4):
            self.foot_positions[:, i] = self._quadruped.getHipLocation(i).reshape(-1) + self._legController.datas[i].p
            self.pFoot[:, i] = self.foot_positions[:, i] + seResult.position

        # * first time initialization
        if self.firstRun:
            self.firstRun = False
            self._stateEstimator._init_contact_history(self.foot_positions)
            for i in range(4):
                self.footSwingTrajectories[i].setHeight(0.15)
                self.footSwingTrajectories[i].setInitialPosition(self.pFoot[:, i])
                self.footSwingTrajectories[i].setFinalPosition(self.pFoot[:, i])

        if flat_ground:
            self._stateEstimator._update_com_position_ground_frame(self.foot_positions)

        # * foot placement
        for leg in range(4):
            self.swingTimes[leg] = self.gait.getCurrentSwingTime(self.dtMPC, leg)

        v_des_robot = np.vstack([self._x_vel_des, self._y_vel_des, self._x_vel_des * 0.0]).T

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

            # pYawCorrected = coordinateRotation(CoordinateAxis.Z, -self._yaw_turn_rate * stance_time / 2) @ pRobotFrame
            pYawCorrected = []
            for j in range(self.batch_size):
                temp = coordinateRotation(CoordinateAxis.Z, -self._yaw_turn_rate[j] * stance_time / 2) @ pRobotFrame
                pYawCorrected.append(temp.flatten())

            pYawCorrected = np.vstack(pYawCorrected)

            Pf = seResult.position + (pYawCorrected + v_des_robot * self.swingTimeRemaining[i])
            pfx_rel = seResult.vBody[:, 0] * (0.5 + cmpc_bonus_swing) * stance_time + \
                      0.03 * (seResult.vBody[:, 0] - v_des_robot[:, 0]) + \
                      (0.5 * seResult.position[:, 2] / 9.81) * (seResult.vBody[:, 1] * self._yaw_turn_rate)

            pfy_rel = seResult.vBody[:, 1] * 0.5 * stance_time * self.dtMPC + \
                      0.03 * (seResult.vBody[:, 1] - v_des_robot[:, 1]) + \
                      (0.5 * seResult.position[:, 2] / 9.81) * (-seResult.vBody[:, 0] * self._yaw_turn_rate)

            p_rel_max = 0.3 * np.ones_like(pfx_rel)

            pfx_rel = np.minimum(np.maximum(pfx_rel, -p_rel_max), p_rel_max)
            pfy_rel = np.minimum(np.maximum(pfy_rel, -p_rel_max), p_rel_max)

            Pf[:, 0] += pfx_rel
            Pf[:, 1] += pfy_rel
            Pf[:, 2] = -0.003

            self.footSwingTrajectories[i].setFinalPosition(Pf)

        # calc gait
        self.iterationCounter += 1

        # gait
        self.contactStates = self.gait.getContactState()
        self.swingStates = self.gait.getSwingState()
        self.mpcTable = self.gait.getMpcTable()

        se_contactState = np.array([0, 0, 0, 0], dtype=DTYPE).reshape((4, 1))

        for foot in range(4):
            contactState = self.contactStates[foot]
            swingState = self.swingStates[foot]
            if swingState > 0:  # * foot is in swing
                if self.firstSwing[foot]:
                    self.firstSwing[foot] = False
                    self.footSwingTrajectories[foot].setInitialPosition(self.pFoot[:, foot])

                self.footSwingTrajectories[foot].computeSwingTrajectoryBezier(swingState, self.swingTimes[foot].item())
                pDesFoot = self.footSwingTrajectories[foot].getPosition()
                vDesFoot = self.footSwingTrajectories[foot].getVelocity()

                pDesLeg = (pDesFoot - seResult.position) \
                          - self._quadruped.getHipLocation(foot).reshape(-1)
                vDesLeg = (vDesFoot - seResult.vBody)

                np.copyto(self._legController.commands[foot].pDes, pDesLeg, casting=CASTING)
                np.copyto(self._legController.commands[foot].vDes, vDesLeg, casting=CASTING)

            else:  # * foot is in stance
                self.firstSwing[foot] = True
                pDesFoot = self.footSwingTrajectories[foot].getPosition()
                vDesFoot = self.footSwingTrajectories[foot].getVelocity()

                pDesLeg = (pDesFoot - seResult.position) \
                          - self._quadruped.getHipLocation(foot).reshape(-1)
                vDesLeg = (vDesFoot - seResult.vBody)

                np.copyto(self._legController.commands[foot].pDes, pDesLeg, casting=CASTING)
                np.copyto(self._legController.commands[foot].vDes, vDesLeg, casting=CASTING)

            se_contactState[foot] = contactState

        self._stateEstimator.setContactPhase(se_contactState)

        # prepare the data to return
        contacts = np.ceil(self._stateEstimator.getContactPhase())
        jacobian = np.concatenate([self._legController.datas[i].J[:, np.newaxis, ...]
                                   for i in range(4)], axis=1)
        foot_pos = np.concatenate([self._legController.datas[i].p[:, np.newaxis, ...]
                                   for i in range(4)], axis=1)
        foot_vel = np.concatenate([self._legController.datas[i].v[:, np.newaxis, ...]
                                   for i in range(4)], axis=1)
        des_foot_pos = np.concatenate([self._legController.commands[i].pDes[:, np.newaxis, ...]
                                       for i in range(4)], axis=1)
        des_foot_vel = np.concatenate([self._legController.commands[i].vDes[:, np.newaxis, ...]
                                       for i in range(4)], axis=1)

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







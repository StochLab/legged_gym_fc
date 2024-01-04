import numpy as np
from legged_gym.envs.go1_fc_new.Go1TrajGenerator.Go1FootSwingTrajectoryParallelNew import FootSwingTrajectory
from legged_gym.envs.go1_fc_new.Go1TrajGenerator.Quadruped import Quadruped
from legged_gym.go1_scripts.go1_utils import getSideSign
from legged_gym.envs.go1_fc_new.Go1TrajGenerator.StateEstimatorParallelNew import StateEstimator
from legged_gym.envs.go1_fc_new.Go1TrajGenerator.LegControllerParallel import LegController
from legged_gym.envs.go1_fc_new.Go1TrajGenerator.math_utils.orientation_tools import coordinateRotation, CoordinateAxis

DTYPE = np.float32
CASTING = "same_kind"
cmpc_bonus_swing = 0.0
flat_ground = True

class OffsetDurationGait:
    """
    trotting, bounding, pronking
    jumping, galloping, standing
    trotRunning, walking, walking2
    pacing
    """
    def __init__(self, nSegment :int, offset :np.ndarray, durations :np.ndarray, name :str):

        # offset in mpc segments
        self.nSegment = nSegment
        self.offsets = offset.flatten()
        # duration of step in mpc segments
        self.durations = durations.flatten()
        # offsets in phase (0 to 1)
        self.offsetsFloat = offset / nSegment
        # durations in phase (0 to 1)
        self.durationsFloat = durations / nSegment
        self.nIterations = nSegment
        self.__name = name
        self._stance = durations[0]
        self._swing = nSegment - durations[0]


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
        self.env_ids = np.arange(self.batch_size, dtype=int)
        self.swing_height = 0.12

        self.firstSwing = np.array([[True for _ in range(4)] for _ in range(self.batch_size)]).T
        self.firstRun = np.array([True] * self.batch_size)
        self.iterationCounter = np.array([0] * self.batch_size)
        self.__phase = np.array([0.0] * self.batch_size)
        self.__iteration = np.array([0.0] * self.batch_size)
        self.__mpc_table = np.zeros((batch_size, self.gait.nSegment * 4))

        self.pFoot = np.zeros((batch_size, 4, 3), dtype=DTYPE)
        self.foot_positions = np.zeros((batch_size, 4, 3), dtype=DTYPE)

        self._quadruped = Quadruped()
        self._legController = LegController(self._quadruped, self.batch_size)
        self._stateEstimator = StateEstimator(self._quadruped, self.batch_size)

        self._x_vel_des = np.array([0.0] * self.batch_size)
        self._y_vel_des = np.array([0.0] * self.batch_size)
        self._yaw_turn_rate = np.array([0.0] * self.batch_size)

        self.contactStates = self.getContactState(self.env_ids).T
        self.swingStates = self.getSwingState(self.env_ids).T
        # self.mpcTable = self.getMpcTable(self.env_ids)

        self.footSwingTrajectories = [FootSwingTrajectory(self.batch_size) for _ in range(4)]
        self.swingTimes = np.zeros(4, dtype=DTYPE)
        self.swingTimeRemaining = np.array([[0.0] * batch_size for _ in range(4)]) # 4 x batch


    def reset(self, env_ids: np.array):
        self.firstRun[env_ids] = True
        self.firstSwing[:, env_ids] = True
        self.iterationCounter[env_ids] = 0.0
        self.__phase[env_ids] = 0.0
        self.__iteration[env_ids] = 0.0
        self.__mpc_table[env_ids] = [0 for _ in range(self.gait.nSegment * 4)]
        self.contactStates[:, env_ids] = self.getContactState(env_ids)[env_ids].T
        self.swingStates[:, env_ids] = self.getSwingState(env_ids)[env_ids].T
        self.__mpc_table[env_ids] = self.getMpcTable(env_ids)[env_ids]
        self.swingTimeRemaining[:, env_ids] = 0

    def setIterations(self, iterationsPerMPC :int, currentIteration :np.array):
        self.__iteration = (currentIteration / iterationsPerMPC) % self.gait.nIterations
        self.__phase = (currentIteration % (iterationsPerMPC * self.gait.nIterations)) / float(iterationsPerMPC * self.gait.nIterations)

    def getContactState(self, env_ids):
        progress = np.zeros((self.batch_size, 4))
        for i in env_ids:
            progress[i] = self.__phase[i] - self.gait.offsetsFloat
            progress[i][progress[i] < 0] += 1.0
            progress[i][progress[i] > self.gait.durationsFloat] = 0.0
            progress[i][progress[i] <= self.gait.durationsFloat] /= self.gait.durationsFloat
        return progress

    def getSwingState(self, env_ids):
        swing_offset = self.gait.offsetsFloat + self.gait.durationsFloat
        for i in range(4):
            if swing_offset[i] > 1:
                swing_offset[i] -= 1.0
        swing_duration = np.ones_like(self.gait.durationsFloat) - self.gait.durationsFloat
        progress = np.zeros((self.batch_size, 4))

        for i in env_ids:
            progress[i] = self.__phase[i] - swing_offset
            progress[i][progress[i] < 0] += 1.0
            progress[i][progress[i] > swing_duration] = 0.0
            progress[i][(progress[i] <= swing_duration) & (swing_duration == 0.0)] = 0.0
            progress[i][(progress[i] <= swing_duration) & (swing_duration != 0.0)] /= swing_duration

        return progress

    def getMpcTable(self, env_ids):
        for j in env_ids:
            for i in range(self.gait.nIterations):
                iter = (i + self.__iteration[j] + 1) % self.gait.nIterations
                progress = iter - self.gait.offsets
                for k in range(4):
                    if progress[k] < 0:
                        progress[k] += self.gait.nIterations
                    if progress[k] < self.gait.durations[k]:
                        self.__mpc_table[j][i * 4 + k] = 1
                    else:
                        self.__mpc_table[j][i * 4 + k] = 0

        return self.__mpc_table

    def getCurrentGaitPhase(self):
        return self.__iteration

    def getCurrentSwingTime(self, dtMPC :float, leg :int):
        return dtMPC * self.gait._swing

    def getCurrentStanceTime(self, dtMPC :float, leg :int):
        return dtMPC * self.gait._stance

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
        # print('seResult pos: ', seResult.position)
        self.setIterations(self.iterationsBetweenMPC, self.iterationCounter)
        self.recomputerTiming(self.default_iterations_between_mpc)

        for i in range(4):
            self.foot_positions[:, i, :] = self._quadruped.getHipLocation(i).reshape(-1) + self._legController.datas[i].p
            self.pFoot[:, i, :] = self.foot_positions[:, i] + seResult.position
            # print('leg pos: ', self.pFoot[:, i].flatten())

        # print('pFoot: ', self.pFoot, self.pFoot.shape)
        indices = np.asarray(np.where(self.firstRun == True)[0])
        if len(indices) > 0:
            self.firstRun[indices] = False
            self._stateEstimator._init_contact_history(indices, self.foot_positions)
            for i in range(4):
                self.footSwingTrajectories[i].setHeight(self.swing_height)
                self.footSwingTrajectories[i].setInitialPosition(indices, self.pFoot[indices, i])
                self.footSwingTrajectories[i].setFinalPosition(indices, self.pFoot[indices, i])


        if flat_ground:
            self._stateEstimator._update_com_position_ground_frame(self.foot_positions)

        # * foot placement
        for leg in range(4):
            self.swingTimes[leg] = self.getCurrentSwingTime(self.dtMPC, leg)

        v_des_robot = np.vstack([self._x_vel_des, self._y_vel_des, self._x_vel_des * 0.0]).T

        for i in range(4):
            indices = np.asarray(np.where(self.firstSwing[i] == True)[0])
            if len(indices) > 0:
                self.swingTimeRemaining[i, indices] = self.swingTimes[i]

            indices = np.asarray(np.where(self.firstSwing[i] == False)[0])
            if len(indices) > 0:
                self.swingTimeRemaining[i, indices] -= self.dt

            self.footSwingTrajectories[i].setHeight(self.swing_height)

            offset = np.array([0, getSideSign(i) * self._quadruped._abadLinkLength, 0], dtype=DTYPE).reshape((3, 1))
            pRobotFrame = self._quadruped.getHipLocation(i) + offset
            stance_time = self.getCurrentStanceTime(self.dtMPC, i)

            pYawCorrected = []
            for j in range(self.batch_size):
                temp = coordinateRotation(CoordinateAxis.Z, -self._yaw_turn_rate[j] * stance_time / 2) @ pRobotFrame
                pYawCorrected.append(temp.flatten())
            pYawCorrected = np.vstack(pYawCorrected)

            Pf = seResult.position + (pYawCorrected + v_des_robot * self.swingTimeRemaining[i].reshape(-1, 1))
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

            # print('leg: ', Pf.flatten())

            self.footSwingTrajectories[i].setFinalPosition(self.env_ids, Pf)

        # calc gait
        self.iterationCounter += 1

        # gait
        self.contactStates = self.getContactState(self.env_ids).T
        self.swingStates = self.getSwingState(self.env_ids).T
        # self.mpcTable = self.getMpcTable(self.env_ids)

        # print('contact states: ', self.contactStates.T[0])
        # print('swing states: ', self.swingStates.T[0])
        # print('pFoot: ', self.pFoot, self.pFoot.shape)

        se_contactState = np.zeros((self.batch_size, 4), dtype=DTYPE)

        for foot in range(4):
            contactState = self.contactStates[foot, :]
            swingState = self.swingStates[foot, :]
            indices = np.asarray(np.where(swingState > 0)[0]) # * foot is in swing
            if len(indices) > 0:
                # print('entered first if', foot)
                env_ids = indices[np.asarray(np.where(self.firstSwing[foot, indices] == True)[0])]
                if len(env_ids) != 0:
                    # print('entered second if', foot)
                    self.firstSwing[foot, env_ids] = False
                    # print('pFoot: ', self.pFoot[env_ids, foot])
                    self.footSwingTrajectories[foot].setInitialPosition(env_ids,
                                                                        self.pFoot[env_ids, foot, :])

                self.footSwingTrajectories[foot].computeSwingTrajectoryBezier(indices,
                                                                              swingState[indices],
                                                                              self.swingTimes[foot])

                pDesFoot = self.footSwingTrajectories[foot].getPosition()[indices]
                vDesFoot = self.footSwingTrajectories[foot].getVelocity()[indices]

                pDesLeg = (pDesFoot - seResult.position[indices]) \
                          - self._quadruped.getHipLocation(foot).reshape(-1)
                vDesLeg = (vDesFoot - seResult.vBody[indices])

                self._legController.commands[foot].pDes[indices, ...] = pDesLeg
                self._legController.commands[foot].vDes[indices, ...] = vDesLeg

            indices = np.asarray(np.where(swingState <= 0)[0])  # * foot is in stance

            if len(indices) > 0:
                self.firstSwing[foot, indices] = True

                pDesFoot = self.footSwingTrajectories[foot].getPosition()[indices]
                vDesFoot = self.footSwingTrajectories[foot].getVelocity()[indices]

                pDesLeg = (pDesFoot - seResult.position[indices]) \
                          - self._quadruped.getHipLocation(foot).reshape(-1)
                vDesLeg = (vDesFoot - seResult.vBody[indices])

                self._legController.commands[foot].pDes[indices, ...] = pDesLeg
                self._legController.commands[foot].vDes[indices, ...] = vDesLeg

            se_contactState[:, foot] = contactState
            self._legController.computeLegAngles(foot)

        self._stateEstimator.setContactPhase(se_contactState)

        # prepare the data to return
        phase = self._stateEstimator.getContactPhase()
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
        des_joint_pos = np.concatenate([self._legController.commands[i].qDes[:, np.newaxis, :]
                                        for i in range(4)], axis=1)

        # mpc_inputs = dict()
        # mpc_inputs['mpcTable'] = self.mpcTable.tolist()
        # mpc_inputs['foot_positions'] = self.foot_positions
        # mpc_inputs['seResult'] = seResult

        swing_outputs = dict()
        swing_outputs['phase'] = phase
        swing_outputs['jacobian'] = jacobian
        swing_outputs['foot_pos'] = foot_pos
        swing_outputs['foot_vel'] = foot_vel
        swing_outputs['des_foot_pos'] = des_foot_pos
        swing_outputs['des_foot_vel'] = des_foot_vel
        swing_outputs['des_joint_pos'] = des_joint_pos

        return swing_outputs
















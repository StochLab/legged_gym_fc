# Class definitions for the kinematics
#
# Created : 17 Feb, 2021
# Modified: 23 Jan, 2023
# Author: Tejas, Aditya Sagi, Aditya Shirwatkar, Shishir, Chandravaran

from typing import Dict, Tuple
import numpy as np
import torch as th
PI = np.pi

class Serial2RKinematics:
    """
    Serial2R Kinematics class
    Functions include : Forward kinematics, inverse kinematics, Jacobian w.r.t the end-effector
    Assumes absolute angles between the links
    """
    def __init__(self, link_lengths: list =[1.0, 1.0]):
        self.link_lengths = link_lengths

    def cosineRule(self, a: np.ndarray, b: np.ndarray, c: np.ndarray)-> np.ndarray:
        '''
        Cosine Rule implementation for triangle sides a, b, c. Each have shape (batch_size, 1)
        cos A
        '''
        return np.arccos(np.clip((c**2 + b**2 - a**2)/(2*b*c), -1, 1))

    def inWorkSpace(self, a: np.ndarray) -> np.ndarray:
        """ 
        Checks if the given points lies inside the workspace space of the serial 2r chain
        Args:
            a: Points to be checked
        Returns:
            mask: A mask specifying which points are inside the workspace
        """
        [l1, l2] = self.link_lengths
        r = np.linalg.norm(a, axis=1)
        return (r**2 > (l1-l2)**2) & (r**2 < (l1+l2)**2)

    def searchSafePositions(self, des_pos: np.ndarray) -> np.ndarray:
        """
        Function to search for the closest end-effector point within the workspace.
        This uses the bisection method to search for a feasible point on the boundary of the workspace.
        
        Args:
            des_pos : desired position of the end-effector
        Return:
            des_pos | p_in : Valid position inside the workspace
        """
        [l1, l2] = self.link_lengths
        rd = (l1 + l2) / 4
        delta_max = 0.001
        max_iter = 20

        r = np.linalg.norm(des_pos, axis=1)
        valid_mask = self.inWorkSpace(des_pos)

        if not valid_mask.any():
            # Bisection method to find the closest point on the boundary
            # Initialize the search space
            p_out = des_pos.copy()
            unit_vec = p_out / np.linalg.norm(p_out, axis=1)[:, np.newaxis]
            p_in = rd * unit_vec

            # Bisection method
            n = 0
            while (np.linalg.norm(p_out - p_in, axis=1) > delta_max).any() and (n < max_iter):
                p = (p_in+p_out)/2
                r = np.linalg.norm(p, axis=1)
                valid_mask = self.inWorkSpace(p)
                p_in[valid_mask] = p[valid_mask]
                p_out[~valid_mask] = p[~valid_mask]
                n += 1

            return p_in
        else:
            return des_pos

    def inverseKinematics(self, ee_pos: np.ndarray, branch=">") -> Tuple[np.ndarray]:
        '''
        Inverse kinematics of a serial 2-R manipulator

        Note - Leg is in x-z plane, rotation about y. And ee_pos has shape (batch_size, 2)
        Args:
            ee_pos: position of the end-effector [x, y] (Cartesian co-ordinates)
            branch: Specify the branch of the leg
        Output:
            valid_mask : A mask specifying which angles are valid
            q : The joint angles of the manipulator [q_hip, q_knee], where the angle q_knee is specified relative to the thigh link
        '''
        Nb, dim = ee_pos.shape
        q = np.zeros((Nb, dim), float)

        [l1, l2] = self.link_lengths
        r = np.linalg.norm(ee_pos, axis=1)

        # Check if the end-effector point lies in the workspace of the manipulator
        valid_mask = self.inWorkSpace(ee_pos)
        invalid_mask = ~valid_mask
        ee_pos[invalid_mask] = self.searchSafePositions(ee_pos[invalid_mask])

        t1 = np.arctan2(-ee_pos[:, 0], -ee_pos[:, 1])

        q[:, 0] = t1 + self.cosineRule(l2, r, l1)
        q[:, 1] = self.cosineRule(r, l1, l2) - PI

        if branch == "<":
            q[:, 0] = t1 - self.cosineRule(l2, r, l1)
            q[:, 1] = q[:, 1] * -1

        return valid_mask, q
    
    def forwardKinematics(self, q: np.ndarray) -> np.ndarray:
        '''
        Forward Kinematics of the serial-2R manipulator

        Note - Leg is in x-z plane, rotation about y. And q has shape (batch_size, 2)

        Args:
            q : A vector of the joint angles [q_hip, q_knee], where q_knee is relative in nature
        Returns:
            p : The position vector of the end-effector
        '''
        [l1, l2] = self.link_lengths
        Nb, dim = q.shape
        p = np.zeros((Nb, dim), float)
        p[:,0] = -l1*np.sin(q[:,0]) + -l2*np.sin(q[:,0:2].sum(axis=1))
        p[:,1] = -l1*np.cos(q[:,0]) + -l2*np.cos(q[:,0:2].sum(axis=1))
        return p

    def Jacobian(self, q: np.ndarray) -> np.ndarray:
        '''
        Provides the Jacobian matrix for the end-effector
        Args:
            q : The joint angles of the manipulator [q_hip, q_knee], where the angle q_knee is specified relative to the thigh link
        Returns:
            mat : A 2x2 velocity Jacobian matrix of the manipulator
        '''
        [l1, l2] = self.link_lengths
        Nb, dim = q.shape
        mat = np.zeros((Nb, dim, dim), float)
        mat[:,0,0] = -l1*np.cos(q[:,0]) - l2*np.cos(q[:,0:2].sum(axis=1))
        mat[:,0,1] = -l2*np.cos(q[:,0:2].sum(axis=1))
        mat[:,1,0] =  l1*np.sin(q[:,0]) + l2*np.sin(q[:,0:2].sum(axis=1))
        mat[:,1,1] =  l2*np.sin(q[:,0:2].sum(axis=1))
        return mat

class Serial3RKinematics:
    def __init__(self, link_lengths: list=[0.5, 1.0, 1.0]):
        self.link_lengths = link_lengths
        self.serial_2R = Serial2RKinematics([link_lengths[1], link_lengths[2]])

    def inWorkSpace(self, a: np.ndarray) -> np.ndarray:
        """ 
        Checks if the given points lies inside the workspace space of the serial 3r chain
        Args:
            a: Points to be checked
        Returns:
            mask: A mask specifying which points are inside the workspace
        """
        [l1, l2, l3] = self.link_lengths
        r = np.linalg.norm(a[:, 1:3], axis=1)
        ee_pos_2r = np.zeros((a.shape[0], 2), float)

        valid = np.ones(a.shape[0], dtype=bool)

        valid1 = (r >= l1)

        ee_pos_2r[:, 0] = a[:, 0]
        ee_pos_2r[valid1, 1] = -np.sqrt(r[valid1]**2 - l1**2)
        ee_pos_2r[~valid1, 1] = -0
        valid2 = self.serial_2R.inWorkSpace(ee_pos_2r)
        
        return valid1 & valid2

    def searchSafePositions(self, des_pos: np.ndarray) -> np.ndarray:
        """
        Function to search for the closest end-effector point within the workspace.
        This uses the bisection method to search for a feasible point on the boundary of the workspace.
        
        Args:
            des_pos : desired position of the end-effector
        Return:
            des_pos | p_in : Valid position inside the workspace
        """
        [l1, l2, l3] = self.link_lengths
        rd = (l1 + l2 + l3) / 6
        delta_max = 0.001
        max_iter = 20

        r = np.linalg.norm(des_pos, axis=1)
        valid_mask = self.inWorkSpace(des_pos)

        if not valid_mask.any():
            # Bisection method to find the closest point on the boundary
            # Initialize the search space
            p_out = des_pos.copy()
            unit_vec = p_out / np.linalg.norm(p_out, axis=1)[:, np.newaxis]
            p_in = rd * unit_vec
            
            # Bisection method
            n = 0
            while (np.linalg.norm(p_out - p_in, axis=1) > delta_max).any() and (n < max_iter):
                p = (p_in+p_out)/2
                r = np.linalg.norm(p, axis=1)
                valid_mask = self.inWorkSpace(p)
                p_in[valid_mask] = p[valid_mask]
                p_out[~valid_mask] = p[~valid_mask]
                n += 1

            return p_in
        else:
            return des_pos

    def inverseKinematics(self, leg_name: str, ee_pos: np.ndarray, branch: str=">") -> Tuple[np.ndarray]:
        '''
        Inverse kinematics of a serial 3-R manipulator

        Note: 
            - Leg is in x-z plane, rotation about y. And ee_pos has shape (batch_size, 3)
            - Note the hip is taken with respective to the negative z axis
            - The solution can be in 2 forms, based on the branch selected 
    
        Args:
            ee_pos: position of the end-effector [x, y, z] (Cartesian co-ordinates)
            branch: Specify the branch of the leg
        Output:
            valid_mask : A mask specifying which angles are valid
            q : The joint angles of the manipulator [q_abd, q_hip, q_knee], where the angle q_knee is specified relative to the thigh link
        '''

        Nb, dim = ee_pos.shape
        q = np.zeros((Nb, dim), float)

        abd_link = self.link_lengths[0]

        valid_mask = self.inWorkSpace(ee_pos)
        ee_pos[~valid_mask] = self.searchSafePositions(ee_pos[~valid_mask])
        l = np.linalg.norm(ee_pos[:, 1:3], axis=1)

        safety_mask = l >= abd_link
        z_prime = np.zeros(safety_mask.shape, dtype=float)
        z_prime[safety_mask] = -np.sqrt(l[safety_mask]**2 - abd_link**2)
        z_prime[~safety_mask] = -0.0

        t1 = np.arctan2(-z_prime, abd_link)

        if leg_name == "FR" or leg_name == "fr" or leg_name == "BR" or leg_name == "br":
            t2 = np.arctan2(-ee_pos[:, 1], -ee_pos[:, 2])
            q[:, 0] = PI/2 - t1 - t2
        else:
            t2 = np.arctan2(ee_pos[:, 1], -ee_pos[:, 2])
            q[:, 0] = t1 + t2 - PI/2

        x_prime = ee_pos[:, 0]
        ee_pos_2r =np.concatenate([x_prime.reshape(-1,1), z_prime.reshape(-1,1)], axis=1)
        valid_mask_2r, q_2r = self.serial_2R.inverseKinematics(ee_pos_2r, branch)
        
        q[:, 1:3] = q_2r
        
        return valid_mask & valid_mask_2r, q

    def forwardKinematics(self, leg_name: str, q: np.ndarray) -> np.ndarray:
        '''
        Forward Kinematics of the serial-3R manipulator

        Note - Leg is in x-z plane, rotation about y. And q has shape (batch_size, 3)

        Args:
            q : A vector of the joint angles [q_abd, q_hip, q_knee], where q_knee is relative in nature
        Returns:
            p : The position vector of the end-effector
        '''

        def rotX(q):
            R = np.zeros((q.shape[0], 3, 3))
            R[:, 0, 0] = 1
            R[:, 0, 1] = 0
            R[:, 0, 2] = 0
            R[:, 1, 0] = 0
            R[:, 1, 1] = np.cos(q)
            R[:, 1, 2] = -np.sin(q)
            R[:, 2, 0] = 0
            R[:, 2, 1] = np.sin(q)
            R[:, 2, 2] = np.cos(q)
            return R

        abd_link = self.link_lengths[0]
        Nb, dim = q.shape
        p = np.zeros((Nb, dim), float)

        q_abd = q[:,0]
        q_2r = np.concatenate([q[:,1].reshape(-1,1), q[:,2].reshape(-1,1)], axis=1)

        v_temp = self.serial_2R.forwardKinematics(q_2r)
        
        if leg_name == "FR" or leg_name == "fr" or leg_name == "BR" or leg_name == "br":
            p[:,1] = -abd_link
        else:
            p[:,1] = abd_link

        p[:,0] = v_temp[:,0]
        p[:,2] = v_temp[:,1]

        # p = np.einsum('ikj,ij->ij', rotX(q_abd), p) # Gives wrong result

        # p = [ R_1 @ p_1 ]
        #     [ R_2 @ p_2 ]
        #     [ ......... ]
        #     [ ......... ]
        #     [ R_N @ p_N ]
        p = np.einsum('ikj,ij->ik', rotX(q_abd), p)

        return p

    def Jacobian(self, leg_name: str, q: np.ndarray) -> np.ndarray:
        """
        Compute the Jacobian of the end effector of leg wrt hip frame (same as base frame)
        Note: q is in batches
        
        Args:
            q : A vector of the joint angles [q_abd, q_hip, q_knee], where q_knee is relative in nature
            leg_name : Name of the leg
        Returns:
            J : Jacobian of size = batch_size x 3 x 3
        """
        if leg_name == "FR" or leg_name == "fr" or leg_name == "BR" or leg_name == "br":
            l1 = -self.link_lengths[0]
        else:
            l1 = self.link_lengths[0]
        
        l2 = self.link_lengths[1]
        l3 = self.link_lengths[2]
        
        s1 = +np.sin(q[:,0]); c1 = +np.cos(q[:,0])
        s2 = +np.sin(q[:,1]); c2 = +np.cos(q[:,1])
        s23 = +np.sin(q[:,1] + q[:,2]); c23 =+np.cos(q[:,1] + q[:,2])

        J = np.zeros((q.shape[0], 3, 3))
        J[:,0,0] = 0
        J[:,0,1] = -l2*c2 - l3*c23
        J[:,0,2] = -l3*c23
        J[:,1,0] = -l1*s1 + l2*c1*c2 + l3*c1*c23
        J[:,1,1] = -l2*s1*s2 - l3*s1*s23
        J[:,1,2] = -l3*s1*s23
        J[:,2,0] = l1*c1 + l2*s1*c2 + l3*s1*c23
        J[:,2,1] = l2*c1*s2 + l3*c1*s23
        J[:,2,2] = l3*c1*s23

        return J

class Stoch3Kinematics(Serial3RKinematics):
    '''
    Class to implement the position and velocity kinematics for the Stoch 3 leg
    Position kinematics: Forward kinematics, Inverse kinematics
    Velocity kinematics: Jacobian
    '''
    def __init__(self, link_parameters: list=[0.123, 0.297 , 0.347], torso_dims: list=[0.541, 0.203, 0.1]):
        Serial3RKinematics.__init__(self, link_parameters)
        self.torso_dims = torso_dims
        self.leg_frames: np.ndarray = np.array(
                                    [[+self.torso_dims[0]/2, +self.torso_dims[1]/2, 0],
                                     [+self.torso_dims[0]/2, -self.torso_dims[1]/2, 0],
                                     [-self.torso_dims[0]/2, +self.torso_dims[1]/2, 0],
                                     [-self.torso_dims[0]/2, -self.torso_dims[1]/2, 0]])
        
        # Not used, will be mostly handled by linear policy
        # self.offsets: np.ndarray = np.array(
        #                             [[+0.00, +0.5, 0],
        #                              [+0.00, -0.5, 0],
        #                              [+0.1, +0.5, 0],
        #                              [+0.1, -0.5, 0]])

        self.offsets: np.ndarray = np.array(
                                    [[+0.00, +0.0, 0],
                                     [+0.00, -0.0, 0],
                                     [+0.00, +0.0, 0],
                                     [+0.00, -0.0, 0]])

    def isaacForwardKinematics(self, q: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        """
        Forward kinematics for all the legs

        Args:
            q : A dictionary of the joint angles ['FL': q_leg, 'FR': q_leg, 'BL': q_leg, 'BR': q_leg], where q_knee is relative in nature
        
        Returns:
            [fl, fr, bl, br] : list of all the end effector positions <x,y,z> for each leg
        """

        fl = self.forwardKinematics("FL", q["FL"].detach().cpu().numpy())  + self.leg_frames[0,:]
        fr = self.forwardKinematics("FR", q["FR"].detach().cpu().numpy())  + self.leg_frames[1,:]
        bl = self.forwardKinematics("BL", q["BL"].detach().cpu().numpy())  + self.leg_frames[2,:]
        br = self.forwardKinematics("BR", q["BR"].detach().cpu().numpy())  + self.leg_frames[3,:]

        return {'FL': th.from_numpy(fl).to(device=q["FL"].device), 
                'FR': th.from_numpy(fr).to(device=q["FR"].device), 
                'BL': th.from_numpy(bl).to(device=q["BL"].device), 
                'BR': th.from_numpy(br).to(device=q["BR"].device)}

    def isaacLegForwardKinematics(self, leg_name: str, q: th.Tensor) -> th.Tensor:
        """
        Forward kinematics for a leg
        """
        r_ee = self.forwardKinematics(leg_name, q.detach().cpu().numpy())
        return th.from_numpy(r_ee).to(device=q.device)

    def getFootVelocity(self, leg_name: str, q: np.ndarray, q_dot: np.ndarray) -> np.ndarray:
        """
        Get Foot velocity of the leg

        Args:
            q : Joint angles of [q_abd, q_hip, q_knee], where q_knee is relative in nature

            q : Joint velocities of [dq_abd, dq_hip, dq_knee]
        
        Returns:
            foot_vel : End effector velocities <vx,vy,vz> the leg
        """

        J = self.Jacobian(leg_name, q)

        # p = [ J_1 @ dq_1 ]
        #     [ J_2 @ dq_2 ]
        #     [ .......... ]
        #     [ .......... ]
        #     [ J_N @ dq_N ]
        foot_vel = np.einsum('ikj,ij->ik', J, q_dot)

        return foot_vel

    def getFootVelocities(self, q: th.Tensor, q_dot: th.Tensor) -> th.Tensor:
        """
        Get Foot velocities for all the legs
        Args:
            q : Joint angles in the order of bl, br, fl, fr
            q_dot : Joint velocities in the same order
        Returns:
            [bl_vel, br_vel, fl_vel, fr_vel] : <vx,vy,vz> for each leg
        """
        device = q.device
        q = q.detach().cpu().numpy()
        q_dot = q_dot.detach().cpu().numpy()

        fl_vel = self.getFootVelocity("FL", q[:, :3], q_dot[:, :3])
        fr_vel = self.getFootVelocity("FR", q[:, 3:6], q_dot[:, 3:6])
        rl_vel = self.getFootVelocity("BL", q[:, 6:9], q_dot[:, 6:9])
        rr_vel = self.getFootVelocity("BR", q[:, 9:12], q_dot[:, 9:12])

        foot_vel = np.concatenate([fl_vel, fr_vel, rl_vel, rr_vel], axis=1)
        foot_vel = th.from_numpy(foot_vel).to(device)
        return foot_vel

    def getIsaacFootVelocities(self, q: Dict[str, th.Tensor], q_dot: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        """
        Get Foot velocities for all the legs

        Args:
            q : A dictionary of the joint angles ['FL': q_leg, 'FR': q_leg, 'BL': q_leg, 'BR': q_leg], where q_knee is relative in nature

            q_dot : A dictionary of the joint velocities ['FL': qd_leg, 'FR': qd_leg, 'BL': qd_leg, 'BR': qd_leg]
        
        Returns:
            [fl_vel, fr_vel, bl_vel, br_vel] : list of all the end effector velocities <vx,vy,vz> for each leg
        """
        fl_vel = self.getFootVelocity("FL", q["FL"].detach().cpu().numpy(), q_dot["FL"].detach().cpu().numpy())
        fr_vel = self.getFootVelocity("FR", q["FR"].detach().cpu().numpy(), q_dot["FR"].detach().cpu().numpy())
        bl_vel = self.getFootVelocity("BL", q["BL"].detach().cpu().numpy(), q_dot["BL"].detach().cpu().numpy())
        br_vel = self.getFootVelocity("BR", q["BR"].detach().cpu().numpy(), q_dot["BR"].detach().cpu().numpy())

        return {'FL': th.from_numpy(fl_vel).to(device=q["FL"].device), 
                'FR': th.from_numpy(fr_vel).to(device=q["FR"].device), 
                'BL': th.from_numpy(bl_vel).to(device=q["BL"].device), 
                'BR': th.from_numpy(br_vel).to(device=q["BR"].device)}

    def isaacJacobian(self, leg_name: str, q: th.Tensor) -> th.Tensor:
        qnp = q.detach().cpu().numpy()
        J = self.Jacobian(leg_name, qnp)
        return th.from_numpy(J).to(dtype=q.dtype, device=q.device)

    def isaacInverseKinematics(self, r: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        """
        Inverse kinematics for all the legs

        Args:
            r : A dictionary of the foot pos
        
        Returns:
            [fl, fr, bl, br] : dict of all joint angles
        """

        _, fl = self.inverseKinematics("FL", r["FL"].detach().cpu().numpy() - self.leg_frames[0,:] + self.offsets[0,:])
        _, fr = self.inverseKinematics("FR", r["FR"].detach().cpu().numpy() - self.leg_frames[1,:] + self.offsets[1,:])
        _, bl = self.inverseKinematics("BL", r["BL"].detach().cpu().numpy() - self.leg_frames[2,:] + self.offsets[2,:])
        _, br = self.inverseKinematics("BR", r["BR"].detach().cpu().numpy() - self.leg_frames[3,:] + self.offsets[3,:])

        return {'FL': th.from_numpy(fl).to(device=r["FL"].device), 
                'FR': th.from_numpy(fr).to(device=r["FR"].device), 
                'BL': th.from_numpy(bl).to(device=r["BL"].device), 
                'BR': th.from_numpy(br).to(device=r["BR"].device)}

    def getTorque(self, leg_name: str, q: th.Tensor, f: th.Tensor) -> th.Tensor:
        """
        Calculates the required torques needed at the joints to achieve the given
        end-effector force at the foot
        Args:
            leg_name: one of [fl, fr, bl, br]
            q: Joint angles of the leg in the following order [abd, hip, knee]
               matrix of shape batch_size x 3
            f: end-effector force needed - batch_size x 3
        Returns:
            torque needed at each joint - batch_size x 3
        """
        jacobian = self.Jacobian(leg_name, q.detach().cpu().numpy())
        # torque = np.einsum('ijk,ik->ij', jacobian, f.detach().cpu().numpy())
        torque = np.einsum('ijk,ij->ik', jacobian, f.detach().cpu().numpy())
        return torque

    def getTorques(self, q: Dict[str, th.Tensor],
                   f: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        """
        Calculates the required torques needed at the joints to achieve the given
        end-effector force at the foot
        Args:
            q: A dictionary of the joint angles ['FL': q_leg, 'FR': q_leg,
            'BL': q_leg, 'BR': q_leg], where q_knee is relative in nature
            f: A dictionary of end-effector forces for each leg
        Returns:
            Dictionary of torques for each leg
        """
        fl_torque = self.getTorque("FL", q["FL"], f["FL"])
        fr_torque = self.getTorque("FR", q["FR"], f["FR"])
        rl_torque = self.getTorque("BL", q["BL"], f["BL"])
        rr_torque = self.getTorque("BR", q["BR"], f["BR"])

        return {'FL': th.from_numpy(fl_torque).to(device=q["FL"].device),
                'FR': th.from_numpy(fr_torque).to(device=q["FR"].device),
                'BL': th.from_numpy(rl_torque).to(device=q["BL"].device),
                'BR': th.from_numpy(rr_torque).to(device=q["BR"].device)}

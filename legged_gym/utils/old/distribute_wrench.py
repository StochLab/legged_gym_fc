import qpSWIFT
import numpy as np
import torch

def hat_operator(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def get_optimal_foot_forces(foot_pos, foot_contacts, desired_dynamics, mu=0.8):
    """
    Distributes the given wrench among the legs s.t friction cone constraints.
    Args:
        foot_pos: foot positions in body frame [bl, br, fl, fr] <x, y, z>
        foot_contacts: True/False for each leg [bl, br, fl, fr]
        desired dynamics: desired wrench on the com of the torso
        mu: friction coefficient
    Returns:
       [bl, br, fl, fr] <x, y, z> Ground Reaction forces for each leg
    """
    eye3 = np.eye(3)
    A_dyna = np.vstack([np.hstack([eye3 * foot_contacts[i] for i in range(4)]),
                        np.hstack([hat_operator(foot_pos[:, i]) * foot_contacts[i]
                                   for i in range(4)])])

    W = np.ones(12)

    '''
    for i in range(4):
        if not foot_contacts[i]:
            W[3*i] = 1e1
            W[3*i+1] = 1e3
            W[3*i+2] = 1e3
    '''

    W = np.diag(W)
    S = 1e8 * np.diag(np.ones(6))
    P = 2 * (A_dyna.T @ S @ A_dyna + W)
    c = 2 * (-A_dyna.T @ S @ desired_dynamics).ravel()
    h = np.zeros(16)

    friction_cone = np.array([[1, 0, -mu],
                              [-1, 0, -mu],
                              [0, 1, -mu],
                              [0, -1, -mu]])

    friction_cone_zero = np.zeros_like(friction_cone)

    G = np.vstack([np.hstack([friction_cone, friction_cone_zero,
                              friction_cone_zero, friction_cone_zero]),
                   np.hstack([friction_cone_zero, friction_cone,
                              friction_cone_zero, friction_cone_zero]),
                   np.hstack([friction_cone_zero, friction_cone_zero,
                              friction_cone, friction_cone_zero]),
                   np.hstack([friction_cone_zero, friction_cone_zero,
                              friction_cone_zero, friction_cone])])

    res = qpSWIFT.run(c, h, P, G)

    # A = A_dyna
    # b = desired_dynamics
    # res = qpSWIFT.run(c,h,P,G,A,b)
    # print(A_dyna @ res['sol'] - desired_dynamics)
    # print("QP time:", (end-start)*1e3)

    return res['sol']


def distribute_wrench(foot_pos: torch.Tensor, foot_contacts: torch.Tensor,
                      wrench: torch.Tensor, mu: float = 0.8) -> torch.Tensor:
    """
    Distributes the wrench among the legs
    Args:
         foot_pos: batch_size x 12
         foot_contacts: batch_size x 4
         wrench: batch_size x 6
         mu: friction coefficient
    Returns:
        Torch tensor of shape batch_size x 12
    """
    batch_size = wrench.shape[0]
    GRFs = []
    device = foot_contacts.device
    foot_pos_np = foot_pos.cpu().numpy()
    foot_contact_np = foot_contacts.cpu().numpy()
    wrench_np = wrench.cpu().numpy()

    for i in range(batch_size):
        grf = get_optimal_foot_forces(foot_pos_np[i, ...].T,
                                      foot_contact_np[i, ...],
                                      wrench_np[i, ...], mu)
        GRFs.append(grf)

    GRFs = np.vstack(GRFs)

    return torch.Tensor(GRFs).to(device)
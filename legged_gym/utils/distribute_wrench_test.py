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

    W = np.diag(W)  # 1e3 * np.diag(W)
    S = 1e8 * np.diag(np.ones(6)) # 1e4 * np.diag(np.ones(6))
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

    options = {'MAXITER': 40, 'RELTOL':1e-4,
               'ABSTOL':1e-4, 'VERBOSE': 0, 'OUTPUT': 3}

    res = qpSWIFT.run(c, h, P, G, opts=options)

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

    if (np.any(np.isnan(foot_pos_np)) or
        np.any(np.isnan(foot_contact_np)) or np.any(np.isnan(wrench_np))):
        print('Nans in inputs to QP!')

    for i in range(batch_size):
        grf = get_optimal_foot_forces(foot_pos_np[i, ...].T,
                                      foot_contact_np[i, ...],
                                      wrench_np[i, ...], mu)
        GRFs.append(grf)

    GRFs = np.vstack(GRFs)

    if np.any(np.isnan(GRFs)):
        print('Nans in QP!')

    GRFs = torch.tensor(GRFs).to(device)

    if torch.any(torch.isnan(GRFs)):
        print('Nans in tensor forces in QP!')

    return GRFs

def main():
    wrench = np.array([[-40.9949, -100.0000, 17.6000, -159.7393, -160.0000, 160.0000],
                       [-100.0000, -100.0000, 17.6000, 160.0000, -160.0000, 160.0000],
                       [-100.0000, -100.0000, 17.6000, 160.0000, -160.0000, 160.0000],
                       [-99.9983, -100.0000, 17.6000, 160.0000, -160.0000, 160.0000]])
    foot_contacts = np.array([[False, True, True, False],
                              [False, True, True, False],
                              [False, True, True, False],
                              [False, True, True, False]])
    foot_pos = np.array([[[0.1857, 0.1257, -0.2214],
                          [0.1861, -0.1339, -0.2944],
                          [-0.1895, 0.1220, -0.2962],
                          [-0.1931, -0.1218, -0.2233]],

                         [[0.1841, 0.1211, -0.2243],
                          [0.1653, -0.1353, -0.2936],
                          [-0.1748, 0.1141, -0.2928],
                          [-0.1924, -0.1228, -0.2214]],

                         [[0.1821, 0.1257, -0.2222],
                          [0.2201, -0.1255, -0.2875],
                          [-0.1854, 0.1174, -0.2968],
                          [-0.1959, -0.1223, -0.2225]],

                         [[0.1866, 0.1217, -0.2235],
                          [0.1773, -0.1320, -0.2956],
                          [-0.1974, 0.1214, -0.2943],
                          [-0.1906, -0.1228, -0.2218]]])
    GRFs = distribute_wrench(torch.tensor(foot_pos, device='cpu'),
                             torch.tensor(foot_contacts, device='cpu'),
                             torch.tensor(wrench, device='cpu'))
    print(GRFs)


if __name__ == '__main__':
    main()

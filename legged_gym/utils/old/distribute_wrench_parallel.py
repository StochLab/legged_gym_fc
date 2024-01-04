import torch
import qpSWIFT


def hat_operator(v, device):
    return torch.tensor([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]], device=device)


def get_optimal_foot_forces(foot_pos, foot_contacts, desired_dynamics, mu=0.8):
    device = foot_pos.device

    eye3 = torch.eye(3, device=device)
    A_dyna = torch.vstack([torch.hstack([eye3 * foot_contacts[i] for i in range(4)]),
                           torch.hstack([hat_operator(foot_pos[:, i], device) * foot_contacts[i]
                                         for i in range(4)])])

    W = torch.ones(12, device=device)

    W = torch.diag(W)
    S = 1e8 * torch.diag(torch.ones(6, device=device))
    P = 2 * (A_dyna.T @ S @ A_dyna + W)
    c = 2 * (-A_dyna.T @ S @ desired_dynamics).view(-1)
    h = torch.zeros(16, device=device)

    friction_cone = torch.tensor([[1, 0, -mu],
                                  [-1, 0, -mu],
                                  [0, 1, -mu],
                                  [0, -1, -mu]], device=device)

    friction_cone_zero = torch.zeros_like(friction_cone)

    G = torch.vstack([torch.hstack([friction_cone, friction_cone_zero,
                                    friction_cone_zero, friction_cone_zero]),
                      torch.hstack([friction_cone_zero, friction_cone,
                                    friction_cone_zero, friction_cone_zero]),
                      torch.hstack([friction_cone_zero, friction_cone_zero,
                                    friction_cone, friction_cone_zero]),
                      torch.hstack([friction_cone_zero, friction_cone_zero,
                                    friction_cone_zero, friction_cone])])

    res = qpSWIFT.run(c.cpu().numpy(), h.cpu().numpy(), P.cpu().numpy(), G.cpu().numpy())

    return torch.tensor(res['sol'], device=device)

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

    for i in range(batch_size):
        grf = get_optimal_foot_forces(foot_pos[i, ...].T.to(device=foot_pos.device),
                                      foot_contacts[i, ...].to(device=foot_pos.device),
                                      wrench[i, ...].to(device=foot_pos.device), mu)
        GRFs.append(grf)

    GRFs = torch.vstack(GRFs)

    return GRFs
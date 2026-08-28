import torch
import torch.nn.functional as F


def weighted_endpoint_loss(logits, p1, mass, gamma=0.5, eps=1e-8):
    log_p = F.log_softmax(logits, dim=-1)
    ce = -(p1 * log_p).sum(dim=-1)
    w = mass.clamp_min(eps) ** gamma
    w = w / w.sum()
    return (w * ce).sum()


def hierarchical_consistency_loss(parent_logits, children_logits, grandchild_mass, eps=1e-8):
    """Cross-level consistency (data.md 19): KL(p_hat_B || p_tilde_B).

    parent_logits   : (B, 8)
    children_logits : (B, 8, 8) row i = prediction of child i of the parent
    grandchild_mass : (B, 8, 8) GT mass of grandchild (i, o); 0 where absent
    """
    p_hat_parent = F.softmax(parent_logits, dim=-1)
    p_hat_children = F.softmax(children_logits, dim=-1)
    est = (p_hat_children * grandchild_mass).sum(dim=-1)
    total = est.sum(dim=-1, keepdim=True)
    valid = (total[:, 0] > eps).float()
    p_tilde = est / total.clamp_min(eps)
    kl = (p_hat_parent * (torch.log(p_hat_parent + eps) - torch.log(p_tilde + eps))).sum(dim=-1)
    denom = valid.sum().clamp_min(1.0)
    return (kl * valid).sum() / denom


def density_and_hierarchy_loss(
    parent_logits,
    children_logits,
    parent_p1,
    parent_mass,
    children_p1,
    children_mass,
    grandchild_mass,
    gamma=0.5,
    lambda_hier=0.1,
):
    B, n = children_logits.shape[0], children_logits.shape[-1]
    logits = torch.cat([parent_logits, children_logits.reshape(B * n, n)], dim=0)
    p1 = torch.cat([parent_p1, children_p1.reshape(B * n, n)], dim=0)
    mass = torch.cat([parent_mass, children_mass.reshape(B * n)], dim=0)
    l_density = weighted_endpoint_loss(logits, p1, mass, gamma=gamma)
    l_hier = hierarchical_consistency_loss(parent_logits, children_logits, grandchild_mass)
    return l_density + lambda_hier * l_hier, l_density.detach(), l_hier.detach()

import torch


def cox_loss(risk_pred, time, event):
    if torch.sum(event) < 1:
        return torch.tensor(0.0, requires_grad=True).to(risk_pred.device)

    order = torch.argsort(time, descending=True)
    risk_pred = risk_pred[order]
    event = event[order]

    risk_exp = torch.exp(risk_pred)
    risk_cumsum = torch.cumsum(risk_exp, dim=0)
    log_risk_cumsum = torch.log(risk_cumsum + 1e-8)

    loss = -torch.sum(event * (risk_pred - log_risk_cumsum)) / (torch.sum(event) + 1e-8)
    return loss


def ratio_loss(gate_probs, min_ratio=0.4, max_ratio=0.6):
    avg_usage = torch.mean(gate_probs[:, 0])
    penalty = torch.clamp(min_ratio - avg_usage, min=0) + torch.clamp(
        avg_usage - max_ratio, min=0
    )
    return penalty**2

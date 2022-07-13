import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.distributions import MultivariateNormal as MVN
import joblib

class GAILossMD(_Loss):
    """
    Multi-Dimension version GAI, compatible with 1-D GAI
    """

    def __init__(self, init_noise_sigma, gmm):
        super(GAILossMD, self).__init__()
        self.gmm = gmm
        self.gmm = {k: torch.tensor(self.gmm[k]) for k in self.gmm}
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma))

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        loss = gai_loss_md(pred, target, self.gmm, noise_var)
        return loss


def gai_loss_md(pred, target, gmm, noise_var):
    I = torch.eye(pred.shape[-1], device=noise_var.device)
    mse_term = -MVN(pred, noise_var*I).log_prob(target)
    balancing_term = MVN(gmm['means'], gmm['variances']+noise_var*I).log_prob(pred.unsqueeze(1)) + gmm['weights'].log()
    balancing_term = torch.logsumexp(balancing_term, dim=1)
    loss = mse_term + balancing_term
    loss = loss * (2 * noise_var).detach()
    return loss.mean()


class BMCLossMD(_Loss):
    """
    Multi-Dimension version BMC, compatible with 1-D BMC
    """

    def __init__(self, init_noise_sigma):
        super(BMCLossMD, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma))
        self.name = "BMCLossMD"

    def forward(self, pred, target, weights=None):
        noise_var = self.noise_sigma ** 2
        loss = bmc_loss_md(pred, target, noise_var)
        return loss


def bmc_loss_md(pred, target, noise_var):
    if noise_var.device != pred.device:
        noise_var = noise_var.to(pred.device)
    I = torch.eye(pred.shape[-1], device=noise_var.device)
    logits = MVN(pred.unsqueeze(1), noise_var*I).log_prob(target.unsqueeze(0))
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0], device=pred.device))
    loss = loss * (2 * noise_var).detach()
    return loss

class GAILoss(_Loss):
    def __init__(self, init_noise_sigma, gmm):
        super(GAILoss, self).__init__()
        self.gmm = joblib.load(gmm)
        self.gmm = {k: torch.tensor(self.gmm[k]).cuda() for k in self.gmm}
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma, device="cuda"))

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        loss = gai_loss(pred, target, self.gmm, noise_var)
        return loss


def gai_loss(pred, target, gmm, noise_var):
    gmm = {k: gmm[k].reshape(1, -1).expand(pred.shape[0], -1) for k in gmm}
    mse_term = F.mse_loss(pred, target, reduction='none') / 2 / noise_var + 0.5 * noise_var.log()
    sum_var = gmm['variances'] + noise_var
    balancing_term = - 0.5 * sum_var.log() - 0.5 * (pred - gmm['means']).pow(2) / sum_var + gmm['weights'].log()
    balancing_term = torch.logsumexp(balancing_term, dim=-1, keepdim=True)
    loss = mse_term + balancing_term
    loss = loss * (2 * noise_var).detach()

    return loss.mean()


class BMCLoss(_Loss):
    def __init__(self, init_noise_sigma):
        super(BMCLoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma, device="cuda"))
        self.name = "BMCLoss"

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        loss = bmc_loss(pred, target, noise_var)
        return loss


def bmc_loss(pred, target, noise_var):
    """Compute the Balanced MSE Loss (BMC) between `pred` and the ground truth `targets`.
    Args:
      pred: A float tensor of size [batch, 1].
      target: A float tensor of size [batch, 1].
      noise_var: A float number or tensor.
    Returns:
      loss: A float tensor. Balanced MSE Loss.
    """
    logits = - (pred - target.T).pow(2) / (2 * noise_var)   # logit size: [batch, batch]
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]))     # contrastive-like loss
    loss = loss * (2 * noise_var).detach()  # optional: restore the loss scale, 'detach' when noise is learnable 

    return loss

class BNILoss(_Loss):
    def __init__(self, init_noise_sigma, bucket_centers, bucket_weights):
        super(BNILoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma, device="cuda"))
        self.bucket_centers = torch.tensor(bucket_centers).cuda()
        self.bucket_weights = torch.tensor(bucket_weights).cuda()

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        loss = bni_loss(pred, target, noise_var, self.bucket_centers, self.bucket_weights)
        return loss

def bni_loss(pred, target, noise_var, bucket_centers, bucket_weights):
    mse_term = F.mse_loss(pred, target, reduction='none') / 2 / noise_var

    num_bucket = bucket_centers.shape[0]
    bucket_center = bucket_centers.unsqueeze(0).repeat(pred.shape[0], 1)
    bucket_weights = bucket_weights.unsqueeze(0).repeat(pred.shape[0], 1)

    balancing_term = - 0.5 * (pred.expand(-1, num_bucket) - bucket_center).pow(2) / noise_var + bucket_weights.log()
    balancing_term = torch.logsumexp(balancing_term, dim=-1, keepdim=True)
    loss = mse_term + balancing_term
    loss = loss * (2 * noise_var).detach()
    return loss.mean()


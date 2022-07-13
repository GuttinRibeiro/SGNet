# Author: Augusto Ribeiro Castro
# Email: augusto.ribeiro.castro@usp.br
# Date: May 31st, 2022

import torch

class MultiTaskLoss(torch.nn.Module):
    '''
        Wraps KLD, CVAE and RMSE loss using a multi task approach to weight
        the contribution of each term rather than naively summing them.
    '''
    def __init__(self, model, eta_traj, eta_obj, eta_kld, goal_criterion, traj_criterion):
        super(MultiTaskLoss, self).__init__()
        self.eta_traj = torch.nn.Parameter(torch.Tensor([eta_traj]))
        self.eta_obj = torch.nn.Parameter(torch.Tensor([eta_obj]))
        self.eta_kld = torch.nn.Parameter(torch.Tensor([eta_kld]))
        self.model = model
        self.goal_criterion = goal_criterion
        self.traj_criterion = traj_criterion
        self.name = 'multitask'

    def forward(self, input, target, weights=None):
        all_goal_traj, cvae_dec_traj, kld_loss, _ = self.model(inputs=input, targets=target)        
        cvae_loss = self.traj_criterion(cvae_dec_traj, target)
        goal_loss = self.goal_criterion(all_goal_traj, target)

        term0 = torch.exp(-self.eta_traj)*cvae_loss + self.eta_traj
        term1 = torch.exp(-self.eta_obj)*goal_loss + self.eta_obj
        term2 = kld_loss.mean()*self.model.param_scheduler.kld_weight

        total_loss = 1/2.0*(term0+term1+term2)
        losses = {
            'total_loss': total_loss,
            'CVAE_loss': cvae_loss,
            'goal_loss': goal_loss,
            'KLD_loss': kld_loss
        }

        return losses, cvae_dec_traj 
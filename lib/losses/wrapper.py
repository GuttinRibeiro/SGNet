# Author: Augusto Ribeiro Castro
# Email: augusto.ribeiro.castro@usp.br
# Date: June 2nd, 2022

import torch

class Wrapper(torch.nn.Module):
    '''
        Wraps KLD, CVAE and RMSE loss using the original loss function proposed by
        SGNet correcting the KLD annealing
    '''
    def __init__(self, model, goal_criterion, traj_criterion, fds=False):
        super(Wrapper, self).__init__()
        self.model = model
        self.goal_criterion = goal_criterion
        self.traj_criterion = traj_criterion
        self.name = 'wrapper'
        self.fds = fds

    def forward(self, input, target, weights, epoch=None):
        if self.fds:
            all_goal_traj, cvae_dec_traj, kld_loss, _, dec_hidden_ret = self.model(inputs=input, targets=target, epoch=epoch)  
        else:
            all_goal_traj, cvae_dec_traj, kld_loss, _ = self.model(inputs=input, targets=target)

        cvae_loss = self.traj_criterion(cvae_dec_traj, target, weights=weights)
        
        if self.goal_criterion.name == "BMCLossMD":
            batch_size = target.shape[0]
            goals = all_goal_traj.view(batch_size, -1)
            goals_gt = target.view(batch_size, -1)
            goal_loss = self.goal_criterion(goals, goals_gt)
        else:
            goal_loss = self.goal_criterion(all_goal_traj, target, weights=weights)

        total_loss = cvae_loss+goal_loss+self.model.param_scheduler.kld_weight*kld_loss.mean()
        losses = {
            'total_loss': total_loss,
            'CVAE_loss': cvae_loss,
            'goal_loss': goal_loss,
            'KLD_loss': kld_loss
        }
        if self.fds:
            return losses, cvae_dec_traj, dec_hidden_ret
        return losses, cvae_dec_traj 
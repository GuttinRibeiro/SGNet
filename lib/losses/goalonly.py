# Author: Augusto Ribeiro Castro
# Email: augusto.ribeiro.castro@usp.br
# Date: June 9th, 2022

import torch

class GoalOnly(torch.nn.Module):
    '''
        Wraps the loss function considering only the goal module.
    '''
    def __init__(self, model):
        super(GoalOnly, self).__init__()
        self.model = model
        self.name = 'goal only'

    def forward(self, input, target, weights=None):
        all_goal_traj, _, _, _ = self.model(inputs=input, targets=target)
        if weights is not None:
            goal_loss = torch.mean(((all_goal_traj[:, -1, :, :] - target[:, -1, :, :])**2)*weights[:, -1, :, :])
        else:
            goal_loss = torch.mean((all_goal_traj[:, -1, :, :] - target[:, -1, :, :])**2)
        # goal_loss = self.goal_criterion(all_goal_traj[:, -1, :, :], target[:, -1, :, :])

        total_loss = goal_loss
        losses = {
            'total_loss': total_loss,
            'CVAE_loss': torch.zeros(1),
            'goal_loss': goal_loss,
            'KLD_loss': torch.zeros(1)
        }

        return losses, all_goal_traj
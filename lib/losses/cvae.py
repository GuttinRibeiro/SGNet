import torch

def get_best_trajectory(pred_traj, target, first_history_index):
    '''Shape: [batch_size, num_observations (enc_step), num_pred, num_traj, bb_coordinates]'''
    with torch.no_grad():
        # Repeat target along its forth dimension
        K = pred_traj.shape[3]
        target = target.unsqueeze(3).repeat(1, 1, 1, K, 1)
        indices = []
        for enc_step in range(first_history_index, pred_traj.size(1)):
            traj_rmse = torch.sqrt(torch.sum((pred_traj[:,enc_step,:,:,:] - target[:,enc_step,:,:,:])**2, dim=-1)).sum(dim=1)
            best_idx = torch.argmin(traj_rmse, dim=1)
            indices.append(best_idx)

    total_loss = []
    for i in range(len(indices)):
        traj_rmse = torch.sqrt(torch.sum((pred_traj[:,i,:,indices[i],:] - target[:,i,:,indices[i],:])**2, dim=-1)).sum(dim=1).mean()
        total_loss.append(traj_rmse)

    return sum(total_loss)/len(total_loss)

def cvae_multi(pred_traj, target, first_history_index = 0):
        '''
        CVAE loss use best-of-many
        '''
        # my_loss = get_best_trajectory(pred_traj, target, first_history_index)
        K = pred_traj.shape[3]
        
        target = target.unsqueeze(3).repeat(1, 1, 1, K, 1)
        total_loss = []
        for enc_step in range(first_history_index, pred_traj.size(1)):
            traj_rmse = torch.sqrt(torch.sum((pred_traj[:,enc_step,:,:,:] - target[:,enc_step,:,:,:])**2, dim=-1)).sum(dim=1)
            best_idx = torch.argmin(traj_rmse, dim=1)
            loss_traj = traj_rmse[range(len(best_idx)), best_idx].mean()
            total_loss.append(loss_traj)

        return sum(total_loss)/len(total_loss)
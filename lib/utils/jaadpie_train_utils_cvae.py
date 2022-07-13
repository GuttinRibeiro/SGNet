import encodings
from tqdm import tqdm
import torch
from torch import nn


from lib.utils.eval_utils import eval_jaad_pie_cvae, show_images, save_images
from lib.losses import cvae_multi

def train(loss, train_gen, optimizer, device, epoch, num_epochs, start_update=0, train_fds_gen=None, writer=None, axis_dict=None, annealing_period='step', fds=False):
    loss.train()
    loss.model.train() # Sets the module in training mode.
    total_goal_loss = 0
    total_cvae_loss = 0
    total_KLD_loss = 0
    loader = tqdm(train_gen, total=len(train_gen))
    loader.set_description("Epoch {}/{}".format(epoch, num_epochs))
    with torch.set_grad_enabled(True):
        for i, data in enumerate(loader):
            batch_size = data['input_x'].shape[0]
            input_traj = data['input_x'].to(device)
            target_traj = data['target_y'].to(device)
            weights = data['weights'].to(device)

            if fds:
                losses, cvae_dec_traj, _ = loss(input_traj, target_traj, weights, epoch=epoch)
            else:
                losses, cvae_dec_traj = loss(input_traj, target_traj, weights)
            
            train_loss = losses['total_loss']
            cvae_loss = losses['CVAE_loss']
            goal_loss = losses['goal_loss']
            KLD_loss = losses['KLD_loss']

            total_goal_loss += goal_loss.item()* batch_size
            total_cvae_loss += cvae_loss.item()* batch_size
            total_KLD_loss += KLD_loss.mean().item()* batch_size

            # optimize
            if annealing_period == 'step':
                loss.model.param_scheduler.step()
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            loader.set_postfix(loss=train_loss.item())
            global_step = epoch*len(train_gen)+i
            if writer is not None:
                writer.add_scalar('train/loss/goal_loss', goal_loss.item(), global_step)
                writer.add_scalar('train/loss/cvae_loss', cvae_loss.item(), global_step)
                writer.add_scalar('train/loss/kld_loss', KLD_loss.mean().item(), global_step)
                writer.add_scalar('train/loss/total_loss', train_loss.item(), global_step)
                writer.add_scalar('train/loss/beta', loss.model.param_scheduler.kld_weight.item(), global_step)
                if loss.name == 'multitask':
                    writer.add_scalar('train/loss/eta_traj', loss.eta_traj.item(), global_step)
                    writer.add_scalar('train/loss/eta_obj', loss.eta_obj.item(), global_step)
                    writer.add_scalar('train/loss/eta_kld', loss.eta_kld.item(), global_step)
                if loss.name == 'wrapper' and loss.goal_criterion.name == "BMCLossMD":
                    writer.add_scalar('train/loss/noise_sigma', loss.goal_criterion.noise_sigma.item(), global_step)

            cvae_dec_traj = cvae_dec_traj.detach().to('cpu').numpy()
            if axis_dict is not None:
                show_images(data, cvae_dec_traj, axis_dict, writer=writer, idx=global_step, token='/train/Image')

    # FDS
    if train_fds_gen is not None and epoch >= start_update:
        encodings, gts = [], []
        print('FDS')
        with torch.set_grad_enabled(False):
            for _, sample_batched in enumerate(tqdm(train_fds_gen)):
                input_traj = sample_batched['input_x'].to(device)
                target_traj = sample_batched['target_y'].to(device)
                weights = sample_batched['weights'].to(device)  
                _, _, feature = loss(input_traj, target_traj, weights, epoch=epoch)
                encodings.append(feature.data.cpu())
                gts.append(target_traj.data.cpu())

        encodings, gts = torch.cat(encodings, 0), torch.cat(gts, 0)
        loss.model.FDS.update_last_epoch_stats(epoch)
        loss.model.FDS.update_running_stats(encodings, gts, epoch)

    total_goal_loss /= len(train_gen.dataset)
    total_cvae_loss/=len(train_gen.dataset)
    total_KLD_loss/=len(train_gen.dataset)
    
    return total_goal_loss, total_cvae_loss, total_KLD_loss

def val(model, val_gen, criterion, device, epoch, num_epochs, writer=None, return_metrics=False, axis_dict=None):
    total_goal_loss = 0
    total_cvae_loss = 0
    total_KLD_loss = 0
    MSE_15 = 0
    MSE_05 = 0 
    MSE_10 = 0 
    FMSE = 0 
    FIOU = 0
    CMSE = 0 
    CFMSE = 0
    model.eval()
    loader = tqdm(val_gen, total=len(val_gen))
    loader.set_description("Epoch {}/{}".format(epoch, num_epochs))
    with torch.set_grad_enabled(False):
        for batch_idx, data in enumerate(loader):
            global_step = epoch*len(val_gen)+batch_idx
            batch_size = data['input_x'].shape[0]
            input_traj = data['input_x'].to(device)
            target_traj = data['target_y'].to(device)

            all_goal_traj, cvae_dec_traj, KLD_loss, _ = model(inputs=input_traj, map_mask=None, targets=None,training=False)

            cvae_loss = cvae_multi(cvae_dec_traj, target_traj)
            if criterion.name == "BMCLossMD":
                batch_size = target_traj.shape[0]
                goals = all_goal_traj.view(batch_size, -1)
                goals_gt = target_traj.view(batch_size, -1)
                goal_loss = criterion(goals, goals_gt)
            else:
                goal_loss = criterion(all_goal_traj, target_traj)

            total_goal_loss += goal_loss.item()* batch_size
            total_cvae_loss += cvae_loss.item()* batch_size
            total_KLD_loss += KLD_loss.mean()* batch_size

            input_traj_np = input_traj.to('cpu').numpy()
            target_traj_np = target_traj.to('cpu').numpy()
            cvae_dec_traj = cvae_dec_traj.to('cpu').numpy()

            if axis_dict is not None:
                show_images(data, cvae_dec_traj, axis_dict, writer=writer, idx=global_step, token='/val/Image')
            batch_MSE_15, batch_MSE_05, batch_MSE_10, batch_FMSE, batch_CMSE, batch_CFMSE, batch_FIOU = eval_jaad_pie_cvae(input_traj_np, target_traj_np[:,-1,:,:], cvae_dec_traj[:,-1,:,:,:])
            
            MSE_15 += batch_MSE_15
            MSE_05 += batch_MSE_05
            MSE_10 += batch_MSE_10
            FMSE += batch_FMSE
            CMSE += batch_CMSE
            CFMSE += batch_CFMSE
            FIOU += batch_FIOU

            val_loss = goal_loss.item() + cvae_loss.item() + KLD_loss.mean().item()
            loader.set_postfix(loss=val_loss)

            if writer is not None:
                writer.add_scalar('val/loss/goal_loss', goal_loss.item(), global_step)
                writer.add_scalar('val/loss/cvae_loss', cvae_loss.item(), global_step)
                writer.add_scalar('val/loss/kld_loss', KLD_loss.mean().item(), global_step)
                writer.add_scalar('val/loss/total_loss', val_loss, global_step)
                writer.add_scalar('val/metric/MSE_05', batch_MSE_05, global_step)
                writer.add_scalar('val/metric/MSE_10', batch_MSE_10, global_step)
                writer.add_scalar('val/metric/MSE_15', batch_MSE_15, global_step)
                writer.add_scalar('val/metric/FMSE', batch_FMSE, global_step)
                writer.add_scalar('val/metric/CMSE', batch_CMSE, global_step)
                writer.add_scalar('val/metric/CFMSE', batch_CFMSE, global_step)
                writer.add_scalar('val/metric/FIOU', batch_FIOU, global_step)

    num_samples = len(val_gen.dataset)
    val_loss = (total_goal_loss+total_cvae_loss+total_KLD_loss)/num_samples

    if return_metrics:
        MSE_15 /= num_samples
        MSE_05 /= num_samples
        MSE_10 /= num_samples
        FMSE /= num_samples
        FIOU /= num_samples
        CMSE /= num_samples
        CFMSE /= num_samples

        if writer is not None:    
            writer.add_scalar('val/loss/epoch/total_loss', val_loss, epoch)
            writer.add_scalar('val/metric/epoch/MSE_05', MSE_05, epoch)
            writer.add_scalar('val/metric/epoch/MSE_10', MSE_10, epoch)
            writer.add_scalar('val/metric/epoch/MSE_15', MSE_15, epoch)
            writer.add_scalar('val/metric/epoch/FMSE', FMSE, epoch)
            writer.add_scalar('val/metric/epoch/CMSE', CMSE, epoch)
            writer.add_scalar('val/metric/epoch/CFMSE', CFMSE, epoch)
            writer.add_scalar('val/metric/epoch/FIOU', FIOU, epoch)

        return val_loss, MSE_15, MSE_05, MSE_10, FMSE, FIOU, CMSE, CFMSE
    return val_loss

def test(model, test_gen, criterion, device, epoch, num_epochs, writer=None, axis_dict=None, save_results=False, save_folder=False):
    total_goal_loss = 0
    total_cvae_loss = 0
    total_KLD_loss = 0
    MSE_15 = 0
    MSE_05 = 0 
    MSE_10 = 0 
    FMSE = 0 
    FIOU = 0
    CMSE = 0 
    CFMSE = 0
    model.eval()
    loader = tqdm(test_gen, total=len(test_gen))
    loader.set_description("Epoch {}/{}".format(epoch, num_epochs))
    with torch.set_grad_enabled(False):
        for batch_idx, data in enumerate(loader):
            global_step = epoch*len(test_gen)+batch_idx
            batch_size = data['input_x'].shape[0]
            input_traj = data['input_x'].to(device)
            target_traj = data['target_y'].to(device)

            all_goal_traj, cvae_dec_traj, KLD_loss, _ = model(inputs=input_traj, map_mask=None, targets=None, training=False)
            cvae_loss = cvae_multi(cvae_dec_traj, target_traj)
            if criterion.name == "BMCLossMD":
                batch_size = target_traj.shape[0]
                goals = all_goal_traj.view(batch_size, -1)
                goals_gt = target_traj.view(batch_size, -1)
                goal_loss = criterion(goals, goals_gt)
            else:
                goal_loss = criterion(all_goal_traj, target_traj)
            test_loss = goal_loss + cvae_loss

            total_goal_loss += goal_loss.item()* batch_size
            total_cvae_loss += cvae_loss.item()* batch_size
            total_KLD_loss += KLD_loss.mean()* batch_size
            input_traj_np = input_traj.to('cpu').numpy()
            target_traj_np = target_traj.to('cpu').numpy()
            cvae_dec_traj = cvae_dec_traj.to('cpu').numpy()

            if save_results:
                save_images(data, cvae_dec_traj, save_folder, multimodal=False)

            if axis_dict is not None:
                show_images(data, cvae_dec_traj, axis_dict, writer=writer, idx=global_step, token='/test/Image')

            batch_MSE_15, batch_MSE_05, batch_MSE_10, batch_FMSE, batch_CMSE, batch_CFMSE, batch_FIOU = eval_jaad_pie_cvae(input_traj_np, target_traj_np[:,-1,:,:], cvae_dec_traj[:,-1,:,:,:])
            MSE_15 += batch_MSE_15
            MSE_05 += batch_MSE_05
            MSE_10 += batch_MSE_10
            FMSE += batch_FMSE
            CMSE += batch_CMSE
            CFMSE += batch_CFMSE
            FIOU += batch_FIOU

            loader.set_postfix(cmse=batch_CMSE)
            if writer is not None:
                writer.add_scalar('test/loss/goal_loss', goal_loss.item(), global_step)
                writer.add_scalar('test/loss/cvae_loss', cvae_loss.item(), global_step)
                writer.add_scalar('test/loss/kld_loss', KLD_loss.mean().item(), global_step)
                test_loss = goal_loss.item()+cvae_loss.item()+KLD_loss.mean().item()
                writer.add_scalar('test/loss/total_loss', test_loss, global_step)
                writer.add_scalar('test/metric/MSE_05', batch_MSE_05, global_step)
                writer.add_scalar('test/metric/MSE_10', batch_MSE_10, global_step)
                writer.add_scalar('test/metric/MSE_15', batch_MSE_15, global_step)
                writer.add_scalar('test/metric/FMSE', batch_FMSE, global_step)
                writer.add_scalar('test/metric/CMSE', batch_CMSE, global_step)
                writer.add_scalar('test/metric/CFMSE', batch_CFMSE, global_step)
                writer.add_scalar('test/metric/FIOU', batch_FIOU, global_step)
    
    num_samples = len(test_gen.dataset)
    MSE_15 /= num_samples
    MSE_05 /= num_samples
    MSE_10 /= num_samples
    FMSE /= num_samples
    FIOU /= num_samples
    CMSE /= num_samples
    CFMSE /= num_samples

    test_loss = (total_goal_loss+total_cvae_loss+total_KLD_loss)/num_samples

    if writer is not None:    
        writer.add_scalar('test/loss/epoch/total_loss', test_loss, epoch)
        writer.add_scalar('test/metric/epoch/MSE_05', MSE_05, epoch)
        writer.add_scalar('test/metric/epoch/MSE_10', MSE_10, epoch)
        writer.add_scalar('test/metric/epoch/MSE_15', MSE_15, epoch)
        writer.add_scalar('test/metric/epoch/FMSE', FMSE, epoch)
        writer.add_scalar('test/metric/epoch/CMSE', CMSE, epoch)
        writer.add_scalar('test/metric/epoch/CFMSE', CFMSE, epoch)
        writer.add_scalar('test/metric/epoch/FIOU', FIOU, epoch)

    return test_loss, MSE_15, MSE_05, MSE_10, FMSE, FIOU, CMSE, CFMSE
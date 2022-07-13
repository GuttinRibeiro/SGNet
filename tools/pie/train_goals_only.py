import os
import os.path as osp
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
from adabelief_pytorch import AdaBelief
import numpy as np
import matplotlib.pyplot as plt

import lib.utils as utl
from configs.pie import parse_sgnet_args as parse_args
from lib.models import build_model
from lib.losses import rmse_loss, GoalOnly
from lib.utils.jaadpie_train_utils_cvae import train

import numpy as np
from tqdm import tqdm
from lib.utils.eval_utils import eval_jaad_pie
from torchinfo import summary

def test(loss, test_gen, device, epoch, num_epochs, writer=None):
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
    loss.model.eval()
    loader = tqdm(test_gen, total=len(test_gen))
    loader.set_description("Epoch {}/{}".format(epoch, num_epochs))
    with torch.set_grad_enabled(False):
        for batch_idx, data in enumerate(loader):
            global_step = epoch*len(test_gen)+batch_idx
            batch_size = data['input_x'].shape[0]
            input_traj = data['input_x'].to(device)
            target_traj = data['target_y'].to(device)

            losses, goals = loss(input_traj, target_traj)
            test_loss = losses['total_loss']
            cvae_loss = losses['CVAE_loss']
            goal_loss = losses['goal_loss']
            KLD_loss = losses['KLD_loss']

            total_goal_loss += goal_loss.item()* batch_size
            total_cvae_loss += cvae_loss.item()* batch_size
            total_KLD_loss += KLD_loss.mean().item()* batch_size

            input_traj_np = input_traj.to('cpu').numpy()
            target_traj_np = target_traj.to('cpu').numpy()
            goals_np = goals.to('cpu').numpy()

            batch_MSE_15, batch_MSE_05, batch_MSE_10, batch_FMSE, batch_CMSE, batch_CFMSE, batch_FIOU = eval_jaad_pie(input_traj_np, target_traj_np[:,-1,:,:], goals_np[:,-1,:,:])
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

def val(loss, val_gen, device, epoch, num_epochs, writer=None, return_metrics=False):
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
    loss.model.eval()
    loader = tqdm(val_gen, total=len(val_gen))
    loader.set_description("Epoch {}/{}".format(epoch, num_epochs))
    with torch.set_grad_enabled(False):
        for batch_idx, data in enumerate(loader):
            global_step = epoch*len(val_gen)+batch_idx
            batch_size = data['input_x'].shape[0]
            input_traj = data['input_x'].to(device)
            target_traj = data['target_y'].to(device)

            losses, goals = loss(input_traj, target_traj)
            val_loss = losses['total_loss']
            cvae_loss = losses['CVAE_loss']
            goal_loss = losses['goal_loss']
            KLD_loss = losses['KLD_loss']

            total_goal_loss += goal_loss.item()* batch_size
            total_cvae_loss += cvae_loss.item()* batch_size
            total_KLD_loss += KLD_loss.mean().item()* batch_size

            input_traj_np = input_traj.to('cpu').numpy()
            target_traj_np = target_traj.to('cpu').numpy()
            goals_np = goals.to('cpu').numpy()

            batch_MSE_15, batch_MSE_05, batch_MSE_10, batch_FMSE, batch_CMSE, batch_CFMSE, batch_FIOU = eval_jaad_pie(input_traj_np, target_traj_np[:,-1,:,:], goals_np[:,-1,:,:])
            
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

def main(args):
    this_dir = osp.dirname(__file__)
    model_name = args.model
    save_dir = osp.join(this_dir, 'checkpoints', model_name, args.experiment, str(args.seed))
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    utl.set_seed(int(args.seed))

    model = build_model(args)
    # summary(model, (args.batch_size, args.enc_steps, 2))
    print('Total number of parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    if osp.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        args.start_epoch += checkpoint['epoch']
        print(f'Checkpoint loaded from file {args.checkpoint}')
 
    model = nn.DataParallel(model)
    model = model.to(device)

    # NOTE: hyperparameter scheduler
    model.param_scheduler = utl.ParamScheduler()
    annealer_kws = {'device': device,
                    'start': args.anneal_start,
                    'finish': args.anneal_finish,
                    'center_step': args.anneal_center_step,
                    'steps_lo_to_hi': args.anneal_steps}
    if args.anneal.lower() == 'sigmoid':
        anneal = utl.sigmoid_anneal
    elif args.anneal.lower() == 'linear': 
        anneal = utl.linear_anneal
    else:
        print(f'Error: {args.anneal} is invalid.')
        exit(1)

    model.param_scheduler.create_new_scheduler(name='kld_weight', annealer=anneal, annealer_kws=annealer_kws)
    loss = GoalOnly(model).to(device)        

    if args.optim.lower() == 'adam':
        print(f'Adam optimizer selected! lr: {args.lr}, wd: {args.weight_decay}')
        optimizer = optim.Adam(loss.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim.lower() == 'sgd':
        print(f'SGD optimizer selected! lr: {args.lr}, wd: {args.weight_decay}')
        optimizer = optim.SGD(loss.parameters(), lr=args.lr, weight_decay=args.weight_decay, nesterov=True, momentum=0.9)
    elif args.optim.lower() == 'adabelief':
        print(f'Adabelief optimizer selected! lr: {args.lr}')
        optimizer = AdaBelief(loss.parameters(), lr=args.lr)   

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=5,
                                                            min_lr=1e-10, verbose=1)

    train_gen = utl.build_data_loader(args, 'train')
    val_gen = utl.build_data_loader(args, 'val')
    test_gen = utl.build_data_loader(args, 'test')
    print("Number of validation samples:", val_gen.__len__())
    print("Number of test samples:", test_gen.__len__())

    if args.disable_tensorboard == False:
        print('Creating tensorboard writer...')
        logdir = 'logs'
        dt_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        writer = SummaryWriter(os.path.join(logdir, "{}, model: {}, seed: {}".format(dt_string, args.model, args.seed)))
        print('Writer created!')
    else:
        writer = None
        print('Tensorboard writer was not created.')

    # Register hyperparameters used in this execution
    hparam_dict = {'model': args.model,
                    'experiment': args.experiment,
                    'batch size': args.batch_size,
                    'early_stopping':  args.early_stopping,
                    'epochs':  args.epochs,
                    'seed': args.seed,
                    'weight decay': args.weight_decay,
                    'learning rate': args.lr,
                    'optimizer': args.optim,
                    'bbox type': args.bbox_type,
                    'normalization': args.normalize,
                    'hidden size': args.hidden_size,
                    'enc steps': args.enc_steps,
                    'dec steps': args.dec_steps,
                    'latent dim': args.LATENT_DIM}  

    # train
    min_MSE_15 = float('inf')
    best_model_metric = None
    num_epochs = args.epochs+args.start_epoch
    print("Number of training samples:", len(train_gen))

    if args.display_images:
        dummy_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        axis_dict = {}
        fig, axs = plt.subplots(2, 2, figsize=(10, 5))
        axis_dict['00'] = axs[0, 0].imshow(dummy_img)
        axis_dict['01'] = axs[0, 1].imshow(dummy_img)
        axis_dict['10'] = axs[1, 0].imshow(dummy_img)
        axis_dict['11'] = axs[1, 1].imshow(dummy_img)
        axis_dict['axs'] = axs
        axis_dict['delay'] = 1/args.display_freq
        axis_dict['fig'] = fig
    else:
        axis_dict=None

    patience = args.patience
    patience_cnt = 0
    for epoch in range(args.start_epoch, num_epochs):
        # train
        train_goal_loss, train_cvae_loss, train_KLD_loss = train(loss, train_gen, optimizer, device, epoch, num_epochs, writer=writer, axis_dict=axis_dict, annealing_period=args.anneal_period.lower())
        writer.add_scalar('train/loss/epoch/goal_loss', train_goal_loss, epoch)
        writer.add_scalar('train/loss/epoch/cvae_loss', train_cvae_loss, epoch)
        writer.add_scalar('train/loss/epoch/kld_loss', train_KLD_loss, epoch)
        print('Train Epoch: {} \t Goal loss: {:.6f}\t CVAE loss: {:.4f}\t KLD loss: {:.4f}'.format(
                epoch, train_goal_loss, train_cvae_loss, train_KLD_loss))

        # val
        val_loss, MSE_15, MSE_05, MSE_10, FMSE, FIOU, CMSE, CFMSE = val(loss, val_gen, device, epoch, num_epochs, return_metrics=True, writer=writer)
        lr_scheduler.step(val_loss)
        print("Validation Loss: {:.6f}".format(val_loss))
        print("MSE_05: %4f;  MSE_10: %4f;  MSE_15: %4f; FMSE: %4f;\n FIOU: %4f; CMSE: %4f; CFMSE: %4f\n" % (MSE_05, MSE_10, MSE_15, FMSE, FIOU, CMSE, CFMSE))

        if MSE_15 < min_MSE_15:
            try:
                os.remove(best_model_metric)
            except:
                pass

            min_MSE_15 = MSE_15
            with open(os.path.join(save_dir, 'metric.txt'),"w") as f:
                f.write("MSE_05: %4f;   MSE_10: %4f;  MSE_15: %4f;  FMSE: %4f;  FIOU: %4f \n" % (MSE_05, MSE_10, min_MSE_15, FMSE, FIOU))
                f.write("CFMSE: %4f;   CMSE: %4f;  \n" % (CFMSE, CMSE))

            saved_model_metric_name = 'best_metric_MSE15_epoch{}.pth'.format(str(format(epoch,'03')))

            print("Saving checkpoints: " + saved_model_metric_name)
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            save_dict = {   'epoch': epoch,
                            'model_state_dict': model.module.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()}
            torch.save(save_dict, os.path.join(save_dir, saved_model_metric_name))
            best_model_metric = os.path.join(save_dir, saved_model_metric_name)
            patience_cnt = 0
        else:
            patience_cnt += 1

        if args.early_stopping and patience_cnt > patience:
            print(f'Training has not improved the model for the last {patience_cnt} epochs. Stopping training...')
            break

    print('Training has finished. Evaluating the best model using the test set: ')

    # test
    # best_model_metric = "tools/pie/checkpoints/SGNet_CVAE/multitask_kld_linear_annealing_multiplying_all_epoch/1/best_metric_MSE15_epoch005.pth"
    model = build_model(args)
    model = model.to(device)   
    checkpoint = torch.load(best_model_metric, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = nn.DataParallel(model)

    print(f'Loaded weights from file {best_model_metric}')

    if args.save_results:
        if not os.path.isdir(args.save_directory):
            os.mkdir(args.save_directory)

    test_loss, t_MSE_15, t_MSE_05, t_MSE_10, t_FMSE, t_FIOU, t_CMSE, t_CFMSE = test(loss, test_gen, device, 0, num_epochs, writer=writer)
    print("Test Loss: {:.4f}".format(test_loss))
    print("MSE_05: %4f;  MSE_10: %4f;  MSE_15: %4f; FMSE: %4f;\n FIOU: %4f; CMSE: %4f; CFMSE: %4f\n" % (t_MSE_05, t_MSE_10, t_MSE_15, t_FMSE, t_FIOU, t_CMSE, t_CFMSE))

    metric_dict = {
        'test/loss/epoch/total_loss': test_loss,
        'test/metric/epoch/MSE_05': t_MSE_05,
        'test/metric/epoch/MSE_10': t_MSE_10,
        'test/metric/epoch/MSE_15': t_MSE_15,
        'test/metric/epoch/FMSE': t_FMSE,
        'test/metric/epoch/FIOU': t_FIOU,
        'test/metric/epoch/CMSE': t_CMSE,
        'test/metric/epoch/CFMSE': t_CFMSE,
    }
    # writer.add_hparams(hparam_dict, metric_dict)  
    writer.add_hparams(hparam_dict, metric_dict, run_name='hparam', timed=False)  
            

if __name__ == '__main__':
    main(parse_args())

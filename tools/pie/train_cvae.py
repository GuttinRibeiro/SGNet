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
from lib.losses import rmse_loss, MultiTaskLoss, Wrapper, cvae_multi, BMCLossMD
from lib.utils.jaadpie_train_utils_cvae import train, val, test

import numpy as np

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
    print('Total number of parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    if osp.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # args.start_epoch += checkpoint['epoch']
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

    print(f'[Balanced MSE] Enabled: {args.bmse}')
    if args.bmse and args.imp == 'bmc':
        print(f'[Balanced MSE] imp: {args.imp}')
        criterion = BMCLossMD(5.0)
    else:
        criterion = rmse_loss().to(device)

    if args.multi_task:
        print('The multi task loss was selected.')
        # multitask_loss = MultiTaskLoss(model, 6.0, 4.0, 1.0, criterion, cvae_multi).to(device)
        loss = MultiTaskLoss(model, 2.0, 2.0, 2.0, criterion, cvae_multi).to(device)        
    else:
        print('The naive loss was selected.')
        loss = Wrapper(model, goal_criterion=criterion, traj_criterion=cvae_multi, fds=args.fds)

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
        logdir = os.path.join('/media', 'olorin', 'Dados', 'logs')
        dt_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        writer = SummaryWriter(os.path.join(logdir, "{}, model: {}, seed: {}".format(dt_string, args.model, args.seed)))
        print('Writer created!')
    else:
        writer = None
        print('Tensorboard writer was not created.')

    # Register hyperparameters used in this execution
    hparam_dict = {'model': args.model,
                    'experiment': args.experiment,
                    'multi task': args.multi_task,
                    'annealer': args.anneal,
                    'annealer_start': args.anneal_start,
                    'annealer_finish': args.anneal_finish,
                    'annealer_center_step': args.anneal_center_step,
                    'annealer_steps': args.anneal_steps,
                    'anneal_period': args.anneal_period,
                    'batch size': args.batch_size,
                    'early_stopping':  args.early_stopping,
                    'epochs':  args.epochs,
                    'seed': args.seed,
                    'weight decay': args.weight_decay,
                    'learning rate': args.lr,
                    'optimizer': args.optim,
                    'bbox type': args.bbox_type,
                    'normalization': args.normalize,
                    'K': args.K,
                    'nu': args.nu,
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
        train_goal_loss, train_cvae_loss, train_KLD_loss = train(loss, train_gen, optimizer, device, epoch, num_epochs, writer=writer, axis_dict=axis_dict, annealing_period=args.anneal_period.lower(), start_update=args.start_update)
        writer.add_scalar('train/loss/epoch/goal_loss', train_goal_loss, epoch)
        writer.add_scalar('train/loss/epoch/cvae_loss', train_cvae_loss, epoch)
        writer.add_scalar('train/loss/epoch/kld_loss', train_KLD_loss, epoch)        
        print('Train Epoch: {} \t Goal loss: {:.4f}\t CVAE loss: {:.4f}\t KLD loss: {:.4f}'.format(
                epoch, train_goal_loss, train_cvae_loss, train_KLD_loss))
        
        if args.anneal_period.lower() == 'epoch' and epoch >= args.increase_beta_after-1:
            model.param_scheduler.step()

        # val
        val_loss, MSE_15, MSE_05, MSE_10, FMSE, FIOU, CMSE, CFMSE = val(model, val_gen, criterion, device, epoch, num_epochs, return_metrics=True, writer=writer, axis_dict=axis_dict)
        lr_scheduler.step(val_loss)
        print("Validation Loss: {:.4f}".format(val_loss))
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
    model = build_model(args)
    model = model.to(device)   
    checkpoint = torch.load(best_model_metric, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = nn.DataParallel(model)

    print(f'Loaded weights from file {best_model_metric}')

    if args.save_results:
        if not os.path.isdir(args.save_directory):
            os.mkdir(args.save_directory)

    test_loss, t_MSE_15, t_MSE_05, t_MSE_10, t_FMSE, t_FIOU, t_CMSE, t_CFMSE = test(model, test_gen, criterion, device, 0, num_epochs, 
                                                                                        writer=writer, axis_dict=axis_dict, save_results=args.save_results, save_folder=args.save_directory)
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
    writer.add_hparams(hparam_dict, metric_dict)              

if __name__ == '__main__':
    main(parse_args())

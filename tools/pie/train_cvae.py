import os
import os.path as osp
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim

import lib.utils as utl
from configs.pie import parse_sgnet_args as parse_args
from lib.models import build_model
from lib.losses import rmse_loss
from lib.utils.jaadpie_train_utils_cvae import train, val, test

def main(args):
    this_dir = osp.dirname(__file__)
    model_name = args.model
    save_dir = osp.join(this_dir, 'checkpoints', model_name, str(args.seed))
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    utl.set_seed(int(args.seed))

    model = build_model(args)
    print('Total number of parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=5,
                                                            min_lr=1e-10, verbose=1)
    if osp.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        args.start_epoch += checkpoint['epoch']

    criterion = rmse_loss().to(device)

    train_gen = utl.build_data_loader(args, 'train')
    val_gen = utl.build_data_loader(args, 'val')
    test_gen = utl.build_data_loader(args, 'test')
    print("Number of validation samples:", val_gen.__len__())
    print("Number of test samples:", test_gen.__len__())

    print('Creating tensorboard writer...')
    logdir = 'logs'
    dt_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(os.path.join(logdir, "{}, model: {}, seed: {}".format(dt_string, args.model, args.seed)))
    print('Writer created!')

    # Register hyperparameters used in this execution
    hparam_dict = {'model': args.model,
                    'batch size': args.batch_size,
                    'epochs':  args.epochs,
                    'seed': args.seed,
                    'weight decay': args.weight_decay,
                    'learning rate': args.lr,
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
    for epoch in range(args.start_epoch, num_epochs):
        print("Number of training samples:", len(train_gen))

        # train
        train_goal_loss, train_cvae_loss, train_KLD_loss = train(model, train_gen, criterion, optimizer, device, epoch, num_epochs, writer=writer)
        print('Train Epoch: {} \t Goal loss: {:.4f}\t CVAE loss: {:.4f}\t KLD loss: {:.4f}'.format(
                epoch, train_goal_loss, train_cvae_loss, train_KLD_loss))

        # val
        val_loss, MSE_15, MSE_05, MSE_10, FMSE, FIOU, CMSE, CFMSE = val(model, val_gen, criterion, device, epoch, num_epochs, return_metrics=True, writer=writer)
        lr_scheduler.step(val_loss)

        # test
        test_loss, t_MSE_15, t_MSE_05, t_MSE_10, t_FMSE, t_FIOU, t_CMSE, t_CFMSE = test(model, test_gen, criterion, device, epoch, num_epochs, writer=writer)
        print("Test Loss: {:.4f}".format(test_loss))
        print("MSE_05: %4f;  MSE_10: %4f;  MSE_15: %4f; FMSE: %4f;\n FIOU: %4f; CMSE: %4f; CFMSE: %4f\n" % (t_MSE_05, t_MSE_10, t_MSE_15, t_FMSE, t_FIOU, t_CMSE, t_CFMSE))

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

            metric_dict = {
                'test/loss': test_loss,
                'test/mse_05': t_MSE_05,
                'test/mse_10': t_MSE_10,
                'test/mse_15': t_MSE_15,
                'test/FMSE': t_FMSE,
                'test/FIOU': t_FIOU,
                'test/CMSE': t_CMSE,
                'test/CFMSE': t_CFMSE,
            }
            writer.add_hparams(hparam_dict, metric_dict)  
            best_model_metric = os.path.join(save_dir, saved_model_metric_name)

if __name__ == '__main__':
    main(parse_args())

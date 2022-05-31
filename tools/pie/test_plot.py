import os
import os.path as osp
from datetime import datetime
from matplotlib import axis
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt

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
    model = model.to(device)

    if osp.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print('Model loaded successfully')

    model = nn.DataParallel(model)
    criterion = rmse_loss().to(device)
    val_gen = utl.build_data_loader(args, 'val')

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

    print('Creating tensorboard writer...')
    logdir = 'logs'
    dt_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(os.path.join(logdir, "{}, model: {}, seed: {}".format(dt_string, args.model, args.seed)))
    print('Writer created!')

    val(model, val_gen, criterion, device, 0, 0, return_metrics=False, axis_dict=axis_dict, writer=writer)


if __name__ == '__main__':
    main(parse_args())
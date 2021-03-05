import argparse
import itertools
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.backends import cudnn
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from model import Backbone, Generator, Discriminator, OSSTCoLoss
from utils import DomainDataset, val

# for reproducibility
np.random.seed(1)
torch.manual_seed(1)
cudnn.deterministic = True
cudnn.benchmark = False


# train for one epoch
def train(net, data_loader, train_optimizer):
    net.train()
    G_content.train()
    G_style.train()
    D_content.train()
    D_style.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader, dynamic_ncols=True)
    for ori_img_1, ori_img_2, pos_index in train_bar:
        ori_img_1, ori_img_2 = ori_img_1.cuda(), ori_img_2.cuda()
        # synthetic domain images
        content = G_content(ori_img_1)
        # shuffle style
        idx = torch.randperm(batch_size, device=ori_img_1.device)
        style = G_style(ori_img_1[idx])
        sytic = content + style

        _, ori_proj_1 = net(ori_img_1)
        # UMDA
        _, ori_proj_2 = net(sytic)
        sim_loss = loss_criterion(ori_proj_1, ori_proj_2)
        content_loss = F.mse_loss(D_content(content), D_content(sytic))
        style_loss = F.mse_loss(D_style(style), D_style(sytic))
        loss = 10 * sim_loss + content_loss + style_loss

        optimizer_G.zero_grad()
        optimizer_D.zero_grad()
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()
        optimizer_G.step()
        optimizer_D.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, gan_epochs, total_loss / total_num))

    return total_loss / total_num


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Model')
    # common args
    parser.add_argument('--data_root', default='data', type=str, help='Datasets root path')
    parser.add_argument('--data_name', default='rgb', type=str, choices=['rgb', 'modal'], help='Dataset name')
    parser.add_argument('--proj_dim', default=128, type=int, help='Projected feature dim for computing loss')
    parser.add_argument('--temperature', default=0.1, type=float, help='Temperature used in softmax')
    parser.add_argument('--z_num', default=8, type=int, help='Number of used styles')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Number of images in each mini-batch for contrast stage')
    parser.add_argument('--gan_epochs', default=1, type=int, help='Number of epoch over the dataset to train gan model')
    parser.add_argument('--contrast_epochs', default=25, type=int,
                        help='Number of epoch over the dataset to train contrast model')
    parser.add_argument('--rounds', default=4, type=int,
                        help='Number of round over the gan model and contrast model to train')
    parser.add_argument('--ranks', default='1,2,4,8', type=str, help='Selected recall')
    parser.add_argument('--save_root', default='result', type=str, help='Result saved root path')

    # args parse
    args = parser.parse_args()
    data_root, data_name, proj_dim, temperature = args.data_root, args.data_name, args.proj_dim, args.temperature
    z_num, batch_size, gan_epochs, contrast_epochs = args.z_num, args.batch_size, args.gan_epochs, args.contrast_epochs
    rounds, save_root, ranks = args.rounds, args.save_root, [int(k) for k in args.ranks.split(',')]

    # data prepare
    train_data = DomainDataset(data_root, data_name, split='train')
    val_data = DomainDataset(data_root, data_name, split='val')
    train_gan_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=8)
    val_gan_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=8)
    train_contrast_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    val_contrast_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=8)

    # model setup
    G_content = Generator(3, 3).cuda()
    G_style = Generator(3, 3).cuda()
    D_content = Discriminator(3).cuda()
    D_style = Discriminator(3).cuda()
    backbone = Backbone(proj_dim).cuda()
    # optimizer config
    optimizer_backbone = Adam(backbone.parameters(), lr=1e-3, weight_decay=1e-6)
    optimizer_G = Adam(itertools.chain(G_content.parameters(), G_style.parameters()), lr=2e-4, betas=(0.5, 0.999))
    optimizer_D_content = Adam(D_content.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_D_style = Adam(D_style.parameters(), lr=2e-4, betas=(0.5, 0.999))
    loss_criterion = OSSTCoLoss(temperature)

    # training loop
    results = {'train_loss': [], 'val_precise': []}
    for rank in ranks:
        if data_name == 'rgb':
            results['val_cf@{}'.format(rank)] = []
            results['val_fr@{}'.format(rank)] = []
            results['val_cr@{}'.format(rank)] = []
            results['val_cross@{}'.format(rank)] = []
        else:
            results['val_cd@{}'.format(rank)] = []
            results['val_dc@{}'.format(rank)] = []
            results['val_cross@{}'.format(rank)] = []
    save_name_pre = '{}_osstco'.format(data_name)
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    best_precise = 0.0
    for epoch in range(1, gan_epochs + 1):
        train_loss = train(backbone, train_gan_loader, optimizer_backbone)
        results['train_loss'].append(train_loss)
        val_precise, features = val(backbone, val_gan_loader, results, ranks, epoch, gan_epochs)
        results['val_precise'].append(val_precise * 100)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('{}/{}_results.csv'.format(save_root, save_name_pre), index_label='epoch')

        if val_precise > best_precise:
            best_precise = val_precise
            torch.save(backbone.state_dict(), '{}/{}_model.pth'.format(save_root, save_name_pre))
            torch.save(features, '{}/{}_vectors.pth'.format(save_root, save_name_pre))

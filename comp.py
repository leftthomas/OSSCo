import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.backends import cudnn
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from model import Backbone, SimCLRLoss, MoCoLoss, NPIDLoss
from utils import DomainDataset, val

# for reproducibility
np.random.seed(1)
torch.manual_seed(1)
cudnn.deterministic = True
cudnn.benchmark = False


# train for one epoch
def train(net, data_loader, train_optimizer):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader, dynamic_ncols=True)
    for img_1, img_2, pos_index in train_bar:
        img_1, img_2 = img_1.cuda(), img_2.cuda()
        _, proj_1 = net(img_1)

        if method_name == 'npid':
            loss, pos_samples = loss_criterion(proj_1, pos_index)
        if method_name == 'simclr':
            _, proj_2 = net(img_2)
            loss = loss_criterion(proj_1, proj_2)
        if method_name == 'moco':
            # shuffle BN
            idx = torch.randperm(batch_size, device=img_2.device)
            _, proj_2 = shadow(img_2[idx])
            proj_2 = proj_2[torch.argsort(idx)]
            loss = loss_criterion(proj_1, proj_2)

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        if method_name == 'npid':
            loss_criterion.enqueue(proj_1, pos_index, pos_samples)
        if method_name == 'moco':
            loss_criterion.enqueue(proj_2)
            # momentum update
            for parameter_q, parameter_k in zip(net.parameters(), shadow.parameters()):
                parameter_k.data.copy_(parameter_k.data * momentum + parameter_q.data * (1.0 - momentum))

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Model')
    # common args
    parser.add_argument('--data_root', default='data', type=str, help='Datasets root path')
    parser.add_argument('--data_name', default='rgb', type=str, choices=['rgb', 'modal'], help='Dataset name')
    parser.add_argument('--method_name', default='simclr', type=str, choices=['simclr', 'moco', 'npid'],
                        help='Method name')
    parser.add_argument('--proj_dim', default=128, type=int, help='Projected feature dim for computing loss')
    parser.add_argument('--temperature', default=0.1, type=float, help='Temperature used in softmax')
    parser.add_argument('--batch_size', default=32, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--iters', default=40000, type=int, help='Number of bp over the model to train')
    parser.add_argument('--ranks', default='1,2,4,8', type=str, help='Selected recall')
    parser.add_argument('--save_root', default='result', type=str, help='Result saved root path')
    # args for NPID and MoCo
    parser.add_argument('--negs', default=4096, type=int, help='Negative sample number')
    parser.add_argument('--momentum', default=0.5, type=float,
                        help='Momentum used for the update of memory bank or shadow model')

    # args parse
    args = parser.parse_args()
    data_root, data_name, method_name = args.data_root, args.data_name, args.method_name
    proj_dim, temperature, batch_size, iters = args.proj_dim, args.temperature, args.batch_size, args.iters
    save_root, negs, momentum = args.save_root, args.negs, args.momentum
    ranks = [int(k) for k in args.ranks.split(',')]

    # data prepare
    train_data = DomainDataset(data_root, data_name, split='train')
    val_data = DomainDataset(data_root, data_name, split='val')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=8)
    # compute the epochs over the dataset
    epochs = iters // (len(train_data) // batch_size)

    # model setup
    model = Backbone(proj_dim).cuda()
    # optimizer config
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    if method_name == 'moco':
        loss_criterion = MoCoLoss(negs, proj_dim, temperature).cuda()
        shadow = Backbone(proj_dim).cuda()
        # initialize shadow as a shadow model of backbone
        for param_q, param_k in zip(model.parameters(), shadow.parameters()):
            param_k.data.copy_(param_q.data)
            # not update by gradient
            param_k.requires_grad = False
    if method_name == 'npid':
        loss_criterion = NPIDLoss(len(train_data), negs, proj_dim, momentum, temperature)
    if method_name == 'simclr':
        loss_criterion = SimCLRLoss(temperature)

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
    save_name_pre = '{}_{}'.format(data_name, method_name)
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    best_precise = 0.0
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer)
        results['train_loss'].append(train_loss)
        val_precise, features = val(model, val_loader, data_name, results, ranks, epoch, epochs)
        results['val_precise'].append(val_precise * 100)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('{}/{}_results.csv'.format(save_root, save_name_pre), index_label='epoch')

        if val_precise > best_precise:
            best_precise = val_precise
            torch.save(model.state_dict(), '{}/{}_model.pth'.format(save_root, save_name_pre))
            torch.save(features, '{}/{}_vectors.pth'.format(save_root, save_name_pre))

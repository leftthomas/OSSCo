import itertools
import os

import pandas as pd
import torch
from torch.nn.functional import mse_loss
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from model import Backbone, Generator, Discriminator, OSSTCoLoss
from utils import DomainDataset, val_contrast, weights_init_normal, ReplayBuffer, parse_common_args


# train for one epoch
def train(net, data_loader, train_optimizer):
    net.train()
    F.train()
    G.train()
    Ds.train()
    D_style.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader, dynamic_ncols=True)
    for ori_img_1, ori_img_2, pos_index in train_bar:
        ori_img_1, ori_img_2 = ori_img_1.cuda(), ori_img_2.cuda()
        # synthetic domain images
        content = F(ori_img_1)
        # shuffle style
        idx = torch.randperm(batch_size, device=ori_img_1.device)
        style = G(ori_img_1[idx])
        sytic = content + style

        _, ori_proj_1 = net(ori_img_1)
        # UMDA
        _, ori_proj_2 = net(sytic)
        sim_loss = criterion_contrast(ori_proj_1, ori_proj_2)
        content_loss = mse_loss(Ds(content), Ds(sytic))
        style_loss = mse_loss(D_style(style), D_style(sytic))
        loss = 10 * sim_loss + content_loss + style_loss

        optimizer_FG.zero_grad()
        optimizer_D.zero_grad()
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()
        optimizer_FG.step()
        optimizer_D.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, gan_epochs, total_loss / total_num))

    return total_loss / total_num


if __name__ == '__main__':
    parser = parse_common_args()
    parser.add_argument('--style_num', default=8, type=int, help='Number of used styles')
    parser.add_argument('--gan_iters', default=4000, type=int, help='Number of bp to train gan model')
    parser.add_argument('--contrast_iters', default=4000, type=int, help='Number of bp to train contrast model')

    # args parse
    args = parser.parse_args()
    data_root, method_name, domains, proj_dim = args.data_root, args.method_name, args.domains, args.proj_dim
    temperature, batch_size, total_iters = args.temperature, args.batch_size, args.total_iters
    style_num, gan_iters, contrast_iters = args.style_num, args.gan_iters, args.contrast_iters
    ranks, save_root, rounds = args.ranks, args.save_root, total_iters // (gan_iters + contrast_iters)
    # asserts
    assert total_iters % (gan_iters + contrast_iters) == 0, \
        'make sure the gan_iters + contrast_iters can be divided by total_iters'
    assert method_name == 'osstco', 'not support for {}'.format(method_name)

    # data prepare
    train_contrast_data = DomainDataset(data_root, domains, split='train')
    train_contrast_loader = DataLoader(train_contrast_data, batch_size=batch_size, shuffle=True, num_workers=8,
                                       drop_last=True)
    val_data = DomainDataset(data_root, domains, split='val')
    val_contrast_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=8)
    val_gan_loader = DataLoader(val_data, batch_size=1, shuffle=False)

    train_gan_data = DomainDataset(data_root, domains, split='train')
    style_images, style_categories, style_labels = train_gan_data.refresh(style_num)
    train_gan_loader = DataLoader(train_gan_data, batch_size=1, shuffle=True)

    # model setup
    F = Generator(4, 3).cuda()
    G = Generator(4, 3).cuda()
    Ds = [Discriminator(3).cuda() for _ in range(z_num)]
    F.apply(weights_init_normal)
    G.apply(weights_init_normal)
    for D in Ds:
        D.apply(weights_init_normal)
    backbone = Backbone(proj_dim).cuda()

    # optimizer config
    optimizer_backbone = Adam(backbone.parameters(), lr=1e-3, weight_decay=1e-6)
    optimizer_FG = Adam(itertools.chain(F.parameters(), G.parameters()), lr=2e-4, betas=(0.5, 0.999))
    optimizer_Ds = [Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999)) for D in Ds]

    # loss setup
    criterion_adversarial = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()
    criterion_contrast = OSSTCoLoss(temperature)

    # training loop
    gan_results = {'train_fg_loss': []}
    for i in range(z_num):
        gan_results['train_d{}_loss'.format(str(i + 1))] = []
    contrast_results = {'train_loss': [], 'val_precise': []}
    for rank in ranks:
        if data_name == 'rgb':
            contrast_results['val_cf@{}'.format(rank)] = []
            contrast_results['val_fr@{}'.format(rank)] = []
            contrast_results['val_cr@{}'.format(rank)] = []
            contrast_results['val_cross@{}'.format(rank)] = []
        else:
            contrast_results['val_cd@{}'.format(rank)] = []
            contrast_results['val_dc@{}'.format(rank)] = []
            contrast_results['val_cross@{}'.format(rank)] = []
    save_name_pre = '{}_osstco'.format(data_name)
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    for i in range(rounds):
        # gan training loop
        fake_buffers = [ReplayBuffer() for _ in range(z_num)]
        for epoch in range(1, gan_epochs + 1):
            g_loss, da_loss, db_loss = train(G_A, G_B, Ds, train_loader, optimizer_G, optimizer_Ds)
            results['train_fg_loss'].append(g_loss)
            results['train_da_loss'].append(da_loss)
            results['train_db_loss'].append(db_loss)
            val_contrast(G_A, G_B, test_loader)
            # save statistics
            data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
            data_frame.to_csv('{}/results.csv'.format(save_root), index_label='epoch')
            torch.save(G_A.state_dict(), '{}/GA.pth'.format(save_root))
            torch.save(G_B.state_dict(), '{}/GB.pth'.format(save_root))
            torch.save(D_A.state_dict(), '{}/DA.pth'.format(save_root))
            torch.save(D_B.state_dict(), '{}/DB.pth'.format(save_root))

        best_precise = 0.0
        for epoch in range(1, gan_epochs + 1):
            train_loss = train(backbone, train_gan_loader, optimizer_backbone)
            contrast_results['train_loss'].append(train_loss)
            val_precise, features = val_contrast(backbone, val_gan_loader, contrast_results, ranks, epoch, gan_epochs)
            contrast_results['val_precise'].append(val_precise * 100)
            # save statistics
            data_frame = pd.DataFrame(data=contrast_results, index=range(1, epoch + 1))
            data_frame.to_csv('{}/{}_results.csv'.format(save_root, save_name_pre), index_label='epoch')

            if val_precise > best_precise:
                best_precise = val_precise
                torch.save(backbone.state_dict(), '{}/{}_model.pth'.format(save_root, save_name_pre))
                torch.save(features, '{}/{}_vectors.pth'.format(save_root, save_name_pre))

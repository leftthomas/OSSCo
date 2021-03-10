import itertools
import os
import random

import pandas as pd
import torch
from PIL import Image
from torch.nn.functional import one_hot
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from model import Backbone, Generator, Discriminator, OSSTCoLoss
from utils import DomainDataset, weights_init_normal, ReplayBuffer, parse_common_args, get_transform, val_contrast

parser = parse_common_args()
parser.add_argument('--style_num', default=8, type=int, help='Number of used styles')
parser.add_argument('--gan_iter', default=4000, type=int, help='Number of bp to train gan model')
parser.add_argument('--rounds', default=5, type=int, help='Number of round to train whole model')

# args parse
args = parser.parse_args()
data_root, method_name, domains, proj_dim = args.data_root, args.method_name, args.domains, args.proj_dim
temperature, batch_size, total_iter = args.temperature, args.batch_size, args.total_iter
style_num, gan_iter, contrast_iter = args.style_num, args.gan_iter, args.total_iter
ranks, save_root, rounds = args.ranks, args.save_root, args.rounds
# asserts
assert method_name == 'osstco', 'not support for {}'.format(method_name)

# data prepare
train_contrast_data = DomainDataset(data_root, domains, split='train')
train_contrast_loader = DataLoader(train_contrast_data, batch_size=batch_size, shuffle=True, num_workers=8,
                                   drop_last=True)
val_data = DomainDataset(data_root, domains, split='val')
val_contrast_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=8)
val_gan_loader = DataLoader(val_data, batch_size=1, shuffle=False)

# model setup
F = Generator(3 + style_num, 3).cuda()
G = Generator(3 + style_num, 3).cuda()
Ds = [Discriminator(3).cuda() for _ in range(style_num)]
F.apply(weights_init_normal)
G.apply(weights_init_normal)
for D in Ds:
    D.apply(weights_init_normal)
backbone = Backbone(proj_dim).cuda()

# optimizer config
optimizer_FG = Adam(itertools.chain(F.parameters(), G.parameters()), lr=2e-4, betas=(0.5, 0.999))
optimizer_Ds = [Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999)) for D in Ds]
optimizer_backbone = Adam(backbone.parameters(), lr=1e-3, weight_decay=1e-6)

# loss setup
criterion_adversarial = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()
criterion_contrast = OSSTCoLoss(temperature)

gan_results = {'train_fg_loss': [], 'train_ds_loss': []}
contrast_results = {'train_loss': [], 'val_precise': []}
save_name_pre = '{}_{}_{}_{}_{}'.format(domains, method_name, style_num, rounds, gan_iter)
if not os.path.exists(save_root):
    os.makedirs(save_root)
best_precise, total_fg_loss, total_ds_loss, total_contrast_loss = 0.0, 0.0, 0.0, 0.0

# training loop
for r in range(1, rounds + 1):
    # each round should refresh style images
    train_gan_data = DomainDataset(data_root, domains, split='train')
    style_images, style_categories, style_labels = train_gan_data.refresh(style_num)
    style_codes = one_hot(torch.arange(0, style_num), style_num).float()
    train_gan_loader = DataLoader(train_gan_data, batch_size=1, shuffle=True)

    fake_style_buffer = [ReplayBuffer() for _ in range(style_num)]
    gan_epochs = (gan_iter // len(train_gan_data)) + 1
    contrast_epochs = (contrast_iter // (len(train_contrast_data) // batch_size)) + 1
    lr_scheduler_FG = LambdaLR(optimizer_FG,
                               lr_lambda=lambda eiter: 1.0 - max(0, eiter - gan_iter // 2) / float(gan_iter // 2))
    lr_scheduler_Ds = [LambdaLR(optimizer_D,
                                lr_lambda=lambda eiter: 1.0 - max(0, eiter - gan_iter // 2) / float(gan_iter // 2))
                       for optimizer_D in optimizer_Ds]
    current_gan_iter, current_contrast_iter = 0, 0

    # GAN training loop
    for epoch in range(1, gan_epochs + 1):
        F.train()
        G.train()
        for D in Ds:
            D.train()
        train_bar = tqdm(train_gan_loader, dynamic_ncols=True)
        for content, _, _, _, _, _ in train_bar:
            content = content.cuda()
            # F and G
            optimizer_FG.zero_grad()
            styles, fake_styles = [], []
            for style, code, D in zip(style_images, style_codes, Ds):
                style = (get_transform('train')(Image.open(style))).unsqueeze(dim=0).cuda()
                styles.append(style)
                code = code.view(1, style_num, 1, 1).cuda()
                code = code.expand(1, style_num, *content.size()[-2:])
                fake_style = F(torch.cat((code, content), dim=1))
                fake_styles.append(fake_style)
                pred_fake_style = D(fake_style)
                # adversarial loss
                target_fake_style = torch.ones(pred_fake_style.size(), device=pred_fake_style.device)
                adversarial_loss = criterion_adversarial(pred_fake_style, target_fake_style)
                # cycle loss
                cycle_loss = criterion_cycle(G(torch.cat((code, fake_style), dim=1)), content)
                # identity loss
                identity_loss = criterion_identity(F(torch.cat((code, style), dim=1)), style) \
                                + criterion_identity(G(torch.cat((code, content), dim=1)), content)
                fg_loss = (2 * adversarial_loss + 20 * cycle_loss + 5 * identity_loss) / style_num
                fg_loss.backward()
                total_fg_loss += fg_loss.item()
            optimizer_FG.step()
            lr_scheduler_FG.step()
            # Ds
            for style, fake_style, fake_buffer, D, optimizer_D, lr_scheduler_D in zip(styles, fake_styles,
                                                                                      fake_style_buffer, Ds,
                                                                                      optimizer_Ds, lr_scheduler_Ds):
                optimizer_D.zero_grad()
                pred_real_style = D(style)
                target_real_style = torch.ones(pred_real_style.size(), device=pred_real_style.device)
                fake_style = fake_buffer.push_and_pop(fake_style)
                pred_fake_style = D(fake_style)
                target_fake_style = torch.zeros(pred_fake_style.size(), device=pred_fake_style.device)
                adversarial_loss = (criterion_adversarial(pred_real_style, target_real_style)
                                    + criterion_adversarial(pred_fake_style, target_fake_style)) / 2
                adversarial_loss.backward()
                optimizer_D.step()
                lr_scheduler_D.step()
                total_ds_loss += adversarial_loss.item() / style_num

            current_gan_iter += 1
            train_bar.set_description('Train Iter: [{}/{}] FG Loss: {:.4f}, Ds Loss: {:.4f}'
                                      .format(current_gan_iter + gan_iter * (r - 1), gan_iter * rounds,
                                              total_fg_loss / (current_gan_iter + gan_iter * (r - 1)),
                                              total_ds_loss / (current_gan_iter + gan_iter * (r - 1))))
            if current_gan_iter % 100 == 0:
                gan_results['train_fg_loss'].append(total_fg_loss / (current_gan_iter + gan_iter * (r - 1)))
                gan_results['train_ds_loss'].append(total_ds_loss / (current_gan_iter + gan_iter * (r - 1)))
                # save statistics
                data_frame = pd.DataFrame(data=gan_results,
                                          index=range(1, (current_gan_iter + gan_iter * (r - 1)) // 100 + 1))
                data_frame.to_csv('{}/{}_gan_results.csv'.format(save_root, save_name_pre), index_label='iter')
            # stop iter data when arriving the gan bp numbers
            if current_gan_iter == gan_iter:
                # save the generated images for val data by current round model and styles
                F.eval()
                with torch.no_grad():
                    for style, category in zip(style_images, style_categories):
                        domain = domains[category]
                        name = os.path.basename(style)
                        style_path = '{}/{}/round-{}/{}_{}'.format(save_root, save_name_pre, r, domain, name)
                        if not os.path.exists(os.path.dirname(style_path)):
                            os.makedirs(os.path.dirname(style_path))
                        Image.open(style).save(style_path)
                    for img, _, img_name, category, label, _ in tqdm(val_gan_loader,
                                                                     desc='Generate images for specific styles',
                                                                     dynamic_ncols=True):
                        for style_image, style_code, style_category in zip(style_images, style_codes, style_categories):
                            style_domain = domains[style_category]
                            style_name = os.path.basename(style_image)
                            domain = domains[category[0]]
                            name = os.path.basename(img_name[0])
                            style_code = style_code.view(1, style_num, 1, 1).cuda()
                            style_code = style_code.expand(1, style_num, *img.size()[-2:])
                            fake_style = (F(torch.cat((style_code, img.cuda()), dim=1)) + 1.0) / 2
                            img_path = '{}/{}/round-{}/{}_{}/{}_{}'.format(save_root, save_name_pre, r, style_domain,
                                                                           style_name.split('.')[0], domain, name)
                            if not os.path.exists(os.path.dirname(img_path)):
                                os.makedirs(os.path.dirname(img_path))
                            save_image(fake_style, img_path)
                F.train()
                # save models
                torch.save(F.state_dict(), '{}/{}_round-{}_F.pth'.format(save_root, save_name_pre, r))
                torch.save(G.state_dict(), '{}/{}_round-{}_G.pth'.format(save_root, save_name_pre, r))
                for i, D in enumerate(Ds):
                    torch.save(D.state_dict(), '{}/{}_round-{}_D{}.pth'.format(save_root, save_name_pre, r, i + 1))
                break
    # contrast training loop
    F.eval()
    for epoch in range(1, contrast_epochs + 1):
        backbone.train()
        train_bar = tqdm(train_contrast_loader, dynamic_ncols=True)
        for img_1, img_2, _, _, _, pos_index in train_bar:
            img_1, img_2 = img_1.cuda(), img_2.cuda()
            _, proj_1 = backbone(img_1)
            _, proj_2 = backbone(img_2)
            with torch.no_grad():
                code = random.choices(torch.chunk(style_codes, style_num, dim=0), k=batch_size)
                code = torch.cat(code, dim=0).view(-1, style_num, 1, 1).cuda()
                code = code.expand(-1, style_num, *img_1.size()[-2:])
                img_3 = F(torch.cat((code, img_1), dim=1))
            _, proj_3 = backbone(img_3)
            loss = criterion_contrast(proj_1, proj_2, proj_3)
            optimizer_backbone.zero_grad()
            loss.backward()
            optimizer_backbone.step()
            current_contrast_iter += 1
            total_contrast_loss += loss.item()
            train_bar.set_description('Train Iter: [{}/{}] Contrast Loss: {:.4f}'
                                      .format(current_contrast_iter + contrast_iter * (r - 1), contrast_iter * rounds,
                                              total_contrast_loss / (current_contrast_iter + contrast_iter * (r - 1))))
            if current_contrast_iter % 100 == 0:
                contrast_results['train_loss'].append(
                    total_contrast_loss / (current_contrast_iter + contrast_iter * (r - 1)))
                # every 100 iters to val the model
                val_precise, features = val_contrast(backbone, val_contrast_loader, contrast_results,
                                                     ranks, current_contrast_iter + contrast_iter * (r - 1),
                                                     contrast_iter * rounds)
                # save statistics
                data_frame = pd.DataFrame(data=contrast_results,
                                          index=range(1, (current_contrast_iter + contrast_iter * (r - 1)) // 100 + 1))
                data_frame.to_csv('{}/{}_results.csv'.format(save_root, save_name_pre), index_label='iter')

                if val_precise > best_precise:
                    best_precise = val_precise
                    torch.save(backbone.state_dict(), '{}/{}_model.pth'.format(save_root, save_name_pre))
                    torch.save(features, '{}/{}_vectors.pth'.format(save_root, save_name_pre))
            # stop iter data when arriving the contrast bp numbers
            if current_contrast_iter == contrast_iter:
                break
    F.train()

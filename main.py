import itertools
import os

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
from utils import DomainDataset, weights_init_normal, ReplayBuffer, parse_common_args, get_transform

parser = parse_common_args()
parser.add_argument('--style_num', default=8, type=int, help='Number of used styles')
parser.add_argument('--gan_iter', default=4000, type=int, help='Number of bp to train gan model')
parser.add_argument('--contrast_iter', default=4000, type=int, help='Number of bp to train contrast model')

# args parse
args = parser.parse_args()
data_root, method_name, domains, proj_dim = args.data_root, args.method_name, args.domains, args.proj_dim
temperature, batch_size, total_iter = args.temperature, args.batch_size, args.total_iter
style_num, gan_iter, contrast_iter = args.style_num, args.gan_iter, args.contrast_iter
ranks, save_root, rounds = args.ranks, args.save_root, total_iter // (gan_iter + contrast_iter)
# asserts
assert total_iter % (gan_iter + contrast_iter) == 0, \
    'make sure the gan_iter + contrast_iter can be divided by total_iter'
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

results = {'train_fg_loss': [], 'train_ds_loss': [], 'train_contrast_loss': [], 'val_precise': []}
save_name_pre = '{}_{}'.format(domains, method_name)
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
            styles = [get_transform('train')(Image.open(style)).cuda() for style in style_images]
            # F and G
            optimizer_FG.zero_grad()
            fake_styles = []
            for i, code in enumerate(style_codes):
                code = code.view(1, style_num, 1, 1).cuda()
                code = code.expand(1, style_num, *content.size()[-2:])
                fake_style = F(torch.cat((code, content), dim=1))
                fake_styles.append(fake_style)
                pred_fake_style = Ds[i](fake_style)
                # adversarial loss
                target_fake_style = torch.ones(pred_fake_style.size(), device=pred_fake_style.device)
                adversarial_loss = criterion_adversarial(pred_fake_style, target_fake_style)
                # cycle loss
                cycle_loss = criterion_cycle(G(torch.cat((code, fake_style), dim=1)), content)
                # identity loss
                identity_loss = criterion_identity(F(torch.cat((code, styles[i]), dim=1)), styles[i]) \
                                + criterion_identity(G(torch.cat((code, content), dim=1)), content)
                fg_loss = (adversarial_loss + 10 * cycle_loss + 5 * identity_loss) / style_num
                fg_loss.backward()
                total_fg_loss += fg_loss.item()
            optimizer_FG.step()
            # Ds
            for i, D in enumerate(Ds):
                optimizer_Ds[i].zero_grad()
                pred_real_style = D(styles[i])
                target_real_style = torch.ones(pred_real_style.size(), device=pred_real_style.device)
                fake_style = fake_style_buffer[i].push_and_pop(fake_styles[i])
                pred_fake_style = D(fake_style)
                target_fake_style = torch.zeros(pred_fake_style.size(), device=pred_fake_style.device)
                adversarial_loss = (criterion_adversarial(pred_real_style, target_real_style)
                                    + criterion_adversarial(pred_fake_style, target_fake_style)) / 2
                adversarial_loss.backward()
                optimizer_Ds[i].step()
                total_ds_loss += adversarial_loss.item() / style_num
            current_gan_iter += 1
            lr_scheduler_FG.step()
            for lr_scheduler_D in lr_scheduler_Ds:
                lr_scheduler_D.step()
            train_bar.set_description('Train Iter: [{}/{}] FG Loss: {:.4f}, Ds Loss: {:.4f}'
                                      .format(current_gan_iter + gan_iter * r, gan_iter * rounds,
                                              total_fg_loss / (current_gan_iter + gan_iter * r),
                                              total_ds_loss / (current_gan_iter + gan_iter * r)))
            if current_gan_iter % 100 == 0:
                results['train_fg_loss'].append(total_fg_loss / (current_gan_iter + gan_iter * r))
                results['train_ds_loss'].append(total_ds_loss / (current_gan_iter + gan_iter * r))
                # save statistics
                data_frame = pd.DataFrame(data=results, index=range(1, (current_gan_iter + gan_iter * r) // 100 + 1))
                data_frame.to_csv('{}/{}_gan_results.csv'.format(save_root, save_name_pre), index_label='iter')
            # stop iter data when arriving the gan bp numbers
            if current_gan_iter == gan_iter:
                # save the generated images for val data by using this round model and styles
                F.eval()
                with torch.no_grad():
                    for i, style in enumerate(style_images):
                        domain = style_categories[i]
                        name = style.split('/')[-1].split('.')[0]
                        style_path = '{}/round-{}/{}_{}_{}.png'.format(save_root, r, i + 1, domain, name)
                        if not os.path.exists(os.path.dirname(save_style_path)):
                            os.makedirs(os.path.dirname(save_style_path))
                        Image.open(style).save(style_path)
                    for img, _, img_name, category, label, _ in tqdm(val_gan_loader, desc='Saving generated images',
                                                                     dynamic_ncols=True):
                        for i, code in enumerate(style_codes):
                            code = code.view(1, style_num, 1, 1).cuda()
                            code = code.expand(1, style_num, *img.size()[-2:])
                            fake_style = (F(torch.cat((code, img), dim=1)) + 1.0) / 2
                            save_style_path = '{}/round-{}/{}'.format(save_root, r, i + 1)
                            if not os.path.exists(save_style_path):
                                os.makedirs(save_style_path)
                            save_image(fake_style, '{}/{}'.format(save_style_path, img_name[0].split('/')[-1]))
                F.train()
                break
# save models
torch.save(F.state_dict(), '{}/{}_F.pth'.format(save_root, save_name_pre))
torch.save(G.state_dict(), '{}/{}_G.pth'.format(save_root, save_name_pre))
for i, D in enumerate(Ds):
    torch.save(D.state_dict(), '{}/{}_D{}.pth'.format(save_root, save_name_pre, i))

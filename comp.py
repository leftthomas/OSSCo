import os

import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from model import Backbone, SimCLRLoss, MoCoLoss, NPIDLoss
from utils import DomainDataset, val_contrast, parse_common_args

parser = parse_common_args()
# args for MoCo and NPID
parser.add_argument('--negs', default=4096, type=int, help='Negative sample number')
parser.add_argument('--momentum', default=0.5, type=float,
                    help='Momentum used for the update of memory bank or shadow model')

# args parse
args = parser.parse_args()
data_root, method_name, domains, proj_dim = args.data_root, args.method_name, args.domains, args.proj_dim
temperature, batch_size, total_iters = args.temperature, args.batch_size, args.total_iters
ranks, save_root, negs, momentum = args.ranks, args.save_root, args.negs, args.momentum

# data prepare
train_data = DomainDataset(data_root, domains, split='train')
val_data = DomainDataset(data_root, domains, split='val')
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=8)

# model setup
model = Backbone(proj_dim).cuda()
# optimizer config
optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
if method_name == 'moco':
    shadow = Backbone(proj_dim).cuda()
    # initialize shadow as a shadow model of backbone
    for param_q, param_k in zip(model.parameters(), shadow.parameters()):
        param_k.data.copy_(param_q.data)
        # not update by gradient
        param_k.requires_grad = False
    loss_criterion = MoCoLoss(negs, proj_dim, temperature).cuda()
if method_name == 'npid':
    loss_criterion = NPIDLoss(len(train_data), negs, proj_dim, momentum, temperature)
if method_name == 'simclr':
    loss_criterion = SimCLRLoss(temperature)
else:
    raise NotImplemented('not support for {}'.format(method_name))

results = {'train_loss': [], 'val_precise': []}
save_name_pre = '{}_{}'.format(domains, method_name)
if not os.path.exists(save_root):
    os.makedirs(save_root)
best_precise, total_loss, current_iters = 0.0, 0.0, 0
epochs = (total_iters // (len(train_data) // batch_size)) + 1

# train loop
for epoch in range(1, epochs + 1):
    model.train()
    train_bar = tqdm(train_loader, dynamic_ncols=True)
    for img_1, img_2, _, _, _, pos_index in train_bar:
        img_1, img_2 = img_1.cuda(), img_2.cuda()
        _, proj_1 = model(img_1)

        if method_name == 'npid':
            loss, pos_samples = loss_criterion(proj_1, pos_index)
        elif method_name == 'simclr':
            _, proj_2 = model(img_2)
            loss = loss_criterion(proj_1, proj_2)
        else:
            # shuffle BN
            idx = torch.randperm(batch_size, device=img_2.device)
            _, proj_2 = shadow(img_2[idx])
            proj_2 = proj_2[torch.argsort(idx)]
            loss = loss_criterion(proj_1, proj_2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if method_name == 'npid':
            loss_criterion.enqueue(proj_1, pos_index, pos_samples)
        if method_name == 'moco':
            loss_criterion.enqueue(proj_2)
            # momentum update
            for parameter_q, parameter_k in zip(model.parameters(), shadow.parameters()):
                parameter_k.data.copy_(parameter_k.data * momentum + parameter_q.data * (1.0 - momentum))

        current_iters += 1
        total_loss += loss.item()
        train_bar.set_description(
            'Train Iters: [{}/{}] Loss: {:.4f}'.format(current_iters, total_iters, total_loss / current_iters))
        if current_iters % 100 == 0:
            results['train_loss'].append(total_loss / current_iters)
            # every 100 iters to val the model
            val_precise, features = val_contrast(model, val_loader, results, ranks, current_iters, total_iters)
            results['val_precise'].append(val_precise * 100)
            # save statistics
            data_frame = pd.DataFrame(data=results, index=range(1, current_iters // 100 + 1))
            data_frame.to_csv('{}/{}_results.csv'.format(save_root, save_name_pre), index_label='iter')

            if val_precise > best_precise:
                best_precise = val_precise
                torch.save(model.state_dict(), '{}/{}_model.pth'.format(save_root, save_name_pre))
                torch.save(features, '{}/{}_vectors.pth'.format(save_root, save_name_pre))
        # stop iter data when arriving the total bp numbers
        if current_iters == total_iters:
            break

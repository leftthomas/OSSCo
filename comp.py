import os

import pandas as pd
import torch
from pytorch_metric_learning.losses import ProxyAnchorLoss, NormalizedSoftmaxLoss
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from model import Backbone, SimCLRLoss, NPIDLoss
from utils import DomainDataset, val_contrast, parse_common_args

parser = parse_common_args()
# args for NPID
parser.add_argument('--negs', default=4096, type=int, help='Negative sample number')
parser.add_argument('--momentum', default=0.5, type=float, help='Momentum used for the update of memory bank')
# args for ProxyAnchor
parser.add_argument('--margin', default=0.1, type=int, help='Margin used for proxy and samples')
parser.add_argument('--alpha', default=32, type=float, help='Scale value used for exponent')

# args parse
args = parser.parse_args()
data_root, data_name, method_name, proj_dim = args.data_root, args.data_name, args.method_name, args.proj_dim
temperature, batch_size, total_iter, margin = args.temperature, args.batch_size, args.total_iter, args.margin
ranks, save_root, negs, momentum, alpha = args.ranks, args.save_root, args.negs, args.momentum, args.alpha

# data prepare
train_data = DomainDataset(data_root, data_name, split='train')
val_data = DomainDataset(data_root, data_name, split='val')
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=8)

# model setup
model = Backbone(proj_dim).cuda()
# optimizer config
optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
if method_name == 'npid':
    loss_criterion = NPIDLoss(len(train_data), negs, proj_dim, momentum, temperature)
elif method_name == 'simclr':
    loss_criterion = SimCLRLoss(temperature)
elif method_name == 'proxyanchor':
    loss_criterion = ProxyAnchorLoss(train_data.num_class, proj_dim, margin, alpha).cuda()
    loss_optimizer = Adam(loss_criterion.parameters(), lr=1e-3, weight_decay=1e-6)
elif method_name == 'normalizedsoftmax':
    loss_criterion = NormalizedSoftmaxLoss(train_data.num_class, proj_dim, temperature).cuda()
    loss_optimizer = Adam(loss_criterion.parameters(), lr=1e-3, weight_decay=1e-6)
else:
    raise NotImplemented('not support for {}'.format(method_name))

results = {'train_loss': [], 'val_precise': []}
save_name_pre = '{}_{}'.format(data_name, method_name)
if not os.path.exists(save_root):
    os.makedirs(save_root)
best_precise, total_loss, current_iter = 0.0, 0.0, 0
epochs = (total_iter // (len(train_data) // batch_size)) + 1

# train loop
for epoch in range(1, epochs + 1):
    model.train()
    train_bar = tqdm(train_loader, dynamic_ncols=True)
    for img_1, img_2, _, _, img_label, pos_index in train_bar:
        img_1, img_2 = img_1.cuda(), img_2.cuda()
        _, proj_1 = model(img_1)

        if method_name == 'npid':
            loss, pos_samples = loss_criterion(proj_1, pos_index)
        elif method_name == 'simclr':
            _, proj_2 = model(img_2)
            loss = loss_criterion(proj_1, proj_2)
        elif method_name == 'proxyanchor':
            loss = loss_criterion(proj_1, img_label)
        else:
            loss = loss_criterion(proj_1, img_label)
        optimizer.zero_grad()
        if method_name in ['proxyanchor', 'normalizedsoftmax']:
            loss_optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if method_name in ['proxyanchor', 'normalizedsoftmax']:
            loss_optimizer.step()

        if method_name == 'npid':
            loss_criterion.enqueue(proj_1, pos_index, pos_samples)

        current_iter += 1
        total_loss += loss.item()
        train_bar.set_description(
            'Train Iter: [{}/{}] Loss: {:.4f}'.format(current_iter, total_iter, total_loss / current_iter))
        if current_iter % 100 == 0:
            results['train_loss'].append(total_loss / current_iter)
            # every 100 iters to val the model
            val_precise, features = val_contrast(model, val_loader, results, ranks, current_iter, total_iter)
            # save statistics
            data_frame = pd.DataFrame(data=results, index=range(1, current_iter // 100 + 1))
            data_frame.to_csv('{}/{}_results.csv'.format(save_root, save_name_pre), index_label='iter')

            if val_precise > best_precise:
                best_precise = val_precise
                torch.save(model.state_dict(), '{}/{}_model.pth'.format(save_root, save_name_pre))
                torch.save(features, '{}/{}_vectors.pth'.format(save_root, save_name_pre))
        # stop iter data when arriving the total bp numbers
        if current_iter == total_iter:
            break

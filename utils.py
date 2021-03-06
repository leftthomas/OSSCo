import argparse
import glob
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.backends import cudnn
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from tqdm import tqdm


def parse_common_args():
    # for reproducibility
    np.random.seed(1)
    torch.manual_seed(1)
    cudnn.deterministic = True
    cudnn.benchmark = False
    # common args
    parser = argparse.ArgumentParser(description='Train Model')
    parser.add_argument('--data_root', default='data', type=str, help='Datasets root path')
    parser.add_argument('--method_name', default='osstco', type=str, choices=['osstco', 'simclr', 'moco', 'npid'],
                        help='Compared method name')
    parser.add_argument('--domains', nargs='+', default=['clear', 'fog', 'rain'], type=str,
                        help='Selected domains to train')
    parser.add_argument('--proj_dim', default=128, type=int, help='Projected feature dim for computing loss')
    parser.add_argument('--temperature', default=0.1, type=float, help='Temperature used in softmax')
    parser.add_argument('--batch_size', default=16, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--total_iters', default=40000, type=int, help='Number of bp to train')
    parser.add_argument('--ranks', nargs='+', default=[1, 2, 4, 8], type=int, help='Selected recall to val')
    parser.add_argument('--save_root', default='result', type=str, help='Result saved root path')
    return parser


def get_transform(split='train'):
    if split == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(256, (1.0, 1.12), interpolation=Image.BICUBIC),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


class DomainDataset(Dataset):
    def __init__(self, data_root, data_name, split='train'):
        super(DomainDataset, self).__init__()

        self.data_name = data_name
        image_path = os.path.join(data_root, data_name, split, '*', '*', '*.png')
        self.images = glob.glob(image_path)
        self.images.sort()
        self.transform = get_transform(split)

    def __getitem__(self, index):
        img_name = self.images[index]
        img = Image.open(img_name)
        img_1 = self.transform(img)
        img_2 = self.transform(img)
        return img_1, img_2, index

    def __len__(self):
        return len(self.images)


def recall(vectors, ranks, data_name):
    if data_name == 'rgb':
        labels = torch.arange(len(vectors) // 3, device=vectors.device)
        labels = torch.cat((labels, labels, labels), dim=0)
        clear_vectors = vectors[:len(vectors) // 3]
        fog_vectors = vectors[len(vectors) // 3: 2 * len(vectors) // 3]
        rain_vectors = vectors[2 * len(vectors) // 3:]
        clear_labels = labels[:len(vectors) // 3]
        fog_labels = labels[len(vectors) // 3: 2 * len(vectors) // 3]
        rain_labels = labels[2 * len(vectors) // 3:]

        # domain clear ---> domain fog
        sim_cf = clear_vectors.mm(fog_vectors.t())
        idx_cf = sim_cf.topk(k=ranks[-1], dim=-1, largest=True)[1]
        # domain fog ---> domain clear
        sim_fc = fog_vectors.mm(clear_vectors.t())
        idx_fc = sim_fc.topk(k=ranks[-1], dim=-1, largest=True)[1]

        # domain clear ---> domain rain
        sim_cr = clear_vectors.mm(rain_vectors.t())
        idx_cr = sim_cr.topk(k=ranks[-1], dim=-1, largest=True)[1]
        # domain rain ---> domain clear
        sim_rc = rain_vectors.mm(clear_vectors.t())
        idx_rc = sim_rc.topk(k=ranks[-1], dim=-1, largest=True)[1]

        # domain fog ---> domain rain
        sim_fr = fog_vectors.mm(rain_vectors.t())
        idx_fr = sim_fr.topk(k=ranks[-1], dim=-1, largest=True)[1]
        # domain rain ---> domain fog
        sim_rf = rain_vectors.mm(fog_vectors.t())
        idx_rf = sim_rf.topk(k=ranks[-1], dim=-1, largest=True)[1]

        # cross domain
        sim = vectors.mm(vectors.t())
        sim.fill_diagonal_(-np.inf)
        idx = sim.topk(k=ranks[-1], dim=-1, largest=True)[1]

        acc = {}
        for r in ranks:
            correct_cf = (torch.eq(fog_labels[idx_cf[:, 0:r]], clear_labels.unsqueeze(dim=-1))).any(dim=-1)
            precise_cf = (torch.sum(correct_cf) / correct_cf.size(0)).item()
            correct_fc = (torch.eq(clear_labels[idx_fc[:, 0:r]], fog_labels.unsqueeze(dim=-1))).any(dim=-1)
            precise_fc = (torch.sum(correct_fc) / correct_fc.size(0)).item()
            acc['cf@{}'.format(r)] = (precise_cf + precise_fc) / 2

            correct_cr = (torch.eq(rain_labels[idx_cr[:, 0:r]], clear_labels.unsqueeze(dim=-1))).any(dim=-1)
            precise_cr = (torch.sum(correct_cr) / correct_cr.size(0)).item()
            correct_rc = (torch.eq(clear_labels[idx_rc[:, 0:r]], rain_labels.unsqueeze(dim=-1))).any(dim=-1)
            precise_rc = (torch.sum(correct_rc) / correct_rc.size(0)).item()
            acc['cr@{}'.format(r)] = (precise_cr + precise_rc) / 2

            correct_fr = (torch.eq(rain_labels[idx_fr[:, 0:r]], fog_labels.unsqueeze(dim=-1))).any(dim=-1)
            precise_fr = (torch.sum(correct_fr) / correct_fr.size(0)).item()
            correct_rf = (torch.eq(fog_labels[idx_rf[:, 0:r]], rain_labels.unsqueeze(dim=-1))).any(dim=-1)
            precise_rf = (torch.sum(correct_rf) / correct_rf.size(0)).item()
            acc['fr@{}'.format(r)] = (precise_fr + precise_rf) / 2

            correct = (torch.eq(labels[idx[:, 0:r]], labels.unsqueeze(dim=-1))).any(dim=-1)
            acc['cross@{}'.format(r)] = (torch.sum(correct) / correct.size(0)).item()
        acc['precise'] = (acc['cf@{}'.format(ranks[0])] + acc['cr@{}'.format(ranks[0])] + acc[
            'fr@{}'.format(ranks[0])] + acc['cross@{}'.format(ranks[0])]) / 4
    else:
        labels = torch.arange(len(vectors) // 2, device=vectors.device)
        labels = torch.cat((labels, labels), dim=0)
        clear_vectors = vectors[:len(vectors) // 2]
        depth_vectors = vectors[len(vectors) // 2:]
        clear_labels = labels[:len(vectors) // 2]
        depth_labels = labels[len(vectors) // 2:]

        # domain clear ---> domain depth
        sim_cd = clear_vectors.mm(depth_vectors.t())
        idx_cd = sim_cd.topk(k=ranks[-1], dim=-1, largest=True)[1]
        # domain depth ---> domain clear
        sim_dc = depth_vectors.mm(clear_vectors.t())
        idx_dc = sim_dc.topk(k=ranks[-1], dim=-1, largest=True)[1]

        # cross domain
        sim = vectors.mm(vectors.t())
        sim.fill_diagonal_(-np.inf)
        idx = sim.topk(k=ranks[-1], dim=-1, largest=True)[1]

        acc = {}
        for r in ranks:
            correct_cd = (torch.eq(depth_labels[idx_cd[:, 0:r]], clear_labels.unsqueeze(dim=-1))).any(dim=-1)
            acc['cd@{}'.format(r)] = (torch.sum(correct_cd) / correct_cd.size(0)).item()
            correct_dc = (torch.eq(clear_labels[idx_dc[:, 0:r]], depth_labels.unsqueeze(dim=-1))).any(dim=-1)
            acc['dc@{}'.format(r)] = (torch.sum(correct_dc) / correct_dc.size(0)).item()
            correct = (torch.eq(labels[idx[:, 0:r]], labels.unsqueeze(dim=-1))).any(dim=-1)
            acc['cross@{}'.format(r)] = (torch.sum(correct) / correct.size(0)).item()
        acc['precise'] = (acc['cd@{}'.format(ranks[0])] + acc['dc@{}'.format(ranks[0])] + acc[
            'cross@{}'.format(ranks[0])]) / 3
    return acc


# val for one epoch
def val_contrast(net, data_loader, results, ranks, epoch, epochs):
    net.eval()
    vectors = []
    with torch.no_grad():
        for data, _, _ in tqdm(data_loader, desc='Feature extracting', dynamic_ncols=True):
            vectors.append(net(data.cuda())[0])
        vectors = torch.cat(vectors, dim=0)
        acc = recall(vectors, ranks, data_loader.dataset.data_name)
        precise = acc['precise']
        desc = 'Val Epoch: [{}/{}] '.format(epoch, epochs)
        for r in ranks:
            if data_loader.dataset.data_name == 'rgb':
                results['val_cf@{}'.format(r)].append(acc['cf@{}'.format(r)] * 100)
                results['val_fr@{}'.format(r)].append(acc['fr@{}'.format(r)] * 100)
                results['val_cr@{}'.format(r)].append(acc['cr@{}'.format(r)] * 100)
                results['val_cross@{}'.format(r)].append(acc['cross@{}'.format(r)] * 100)
            else:
                results['val_cd@{}'.format(r)].append(acc['cd@{}'.format(r)] * 100)
                results['val_dc@{}'.format(r)].append(acc['dc@{}'.format(r)] * 100)
                results['val_cross@{}'.format(r)].append(acc['cross@{}'.format(r)] * 100)
        if data_loader.dataset.data_name == 'rgb':
            desc += '| (C<->F) R@{}:{:.2f}% | '.format(ranks[0], acc['cf@{}'.format(ranks[0])] * 100)
            desc += '(F<->R) R@{}:{:.2f}% | '.format(ranks[0], acc['fr@{}'.format(ranks[0])] * 100)
            desc += '(C<->R) R@{}:{:.2f}% | '.format(ranks[0], acc['cr@{}'.format(ranks[0])] * 100)
            desc += '(Cross) R@{}:{:.2f}% | '.format(ranks[0], acc['cross@{}'.format(ranks[0])] * 100)
        else:
            desc += '| (C->D) R@{}:{:.2f}% | '.format(ranks[0], acc['cd@{}'.format(ranks[0])] * 100)
            desc += '(D->C) R@{}:{:.2f}% | '.format(ranks[0], acc['dc@{}'.format(ranks[0])] * 100)
            desc += '(C<->D) R@{}:{:.2f}% | '.format(ranks[0], acc['cross@{}'.format(ranks[0])] * 100)
        print(desc)
    net.train()
    return precise, vectors


class ReplayBuffer:
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.detach():
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0.0)

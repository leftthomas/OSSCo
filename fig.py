import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from utils import DomainDataset


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    colors1 = '#00CED1'
    colors2 = '#DC143C'
    for i in range(data.shape[0]):
        if label[i]:
            colors = colors1
        else:
            colors = colors2
        plt.scatter(data[i, 0], data[i, 1], s=10, c=colors)
    plt.savefig('result/{}.pdf'.format(title), dpi=30, bbox_inches='tight', pad_inches=0)


data_name = 'cufsf'

val_data = DomainDataset('data', data_name, split='val')
pretrained_vectors = torch.load('result/{}_pretrained_vectors.pth'.format(data_name)).cpu().numpy()
npid_vectors = torch.load('result/{}_npid_vectors.pth'.format(data_name)).cpu().numpy()
simclr_vectors = torch.load('result/{}_simclr_vectors.pth'.format(data_name)).cpu().numpy()
proxyanchor_vectors = torch.load('result/{}_proxyanchor_vectors.pth'.format(data_name)).cpu().numpy()
softtriple_vectors = torch.load('result/{}_softtriple_vectors.pth'.format(data_name)).cpu().numpy()
ossco_vectors = torch.load('result/{}_ossco_vectors.pth'.format(data_name)).cpu().numpy()

labels = torch.cat((torch.ones(len(simclr_vectors) // 2, dtype=torch.long),
                    torch.zeros(len(simclr_vectors) // 2, dtype=torch.long)), dim=0).cpu().numpy()

tsne = TSNE(n_components=2, init='pca', random_state=0)

pretrained_results = tsne.fit_transform(pretrained_vectors)
plot_embedding(pretrained_results, labels, 'pretrained_{}'.format(data_name))
npid_results = tsne.fit_transform(npid_vectors)
plot_embedding(npid_results, labels, 'npid_{}'.format(data_name))
simclr_results = tsne.fit_transform(simclr_vectors)
plot_embedding(simclr_results, labels, 'simclr_{}'.format(data_name))
proxyanchor_results = tsne.fit_transform(proxyanchor_vectors)
plot_embedding(proxyanchor_results, labels, 'proxyanchor_{}'.format(data_name))
softtriple_results = tsne.fit_transform(softtriple_vectors)
plot_embedding(softtriple_results, labels, 'softtriple_{}'.format(data_name))
ossco_results = tsne.fit_transform(ossco_vectors)
plot_embedding(ossco_results, labels, 'ossco_{}'.format(data_name))

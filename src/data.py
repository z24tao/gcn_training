import torch

base_path = '../data'


def load(dataset, epoch, name, dense=True):
    path = f'{base_path}/{dataset}/{epoch}/{name}.pt'
    x = torch.load(path, map_location=torch.device('cpu'))
    if dense:
        return x.numpy()
    x = x.coalesce()
    xi = x.indices().numpy()
    xv = x.values().numpy()
    return xi, xv

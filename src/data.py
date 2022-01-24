import csv
import torch

base_path = '../data'


def load(dataset, epoch, name, dtype, dense=True):
    path = f'{base_path}/{dataset}/{epoch}/{name}.pt'
    x = torch.load(path, map_location=torch.device('cpu'))
    if dense:
        return x.numpy().astype(dtype)
    x = x.coalesce()
    xi = x.indices().numpy()
    xv = x.values().numpy().astype(dtype)
    return xi, xv


def write_file(data, fn, line_length=128):
    writer = csv.writer(open(fn + '.txt', 'w'), delimiter=' ')
    for start in range(0, len(data), line_length):
        writer.writerow([data[start:start + line_length]])

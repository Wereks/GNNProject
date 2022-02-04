import rdkit, rdkit.Chem
import numpy as np
import pandas as pd
import torch

ATOMS = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'}


def smiles2graph(sml):
    m = rdkit.Chem.MolFromSmiles(sml)
    m = rdkit.Chem.AddHs(m)

    N = len(list(m.GetAtoms()))
    nodes = np.zeros((N, len(ATOMS)))
    lookup = list(ATOMS.keys())

    for i in m.GetAtoms():
        nodes[i.GetIdx(), lookup.index(i.GetAtomicNum())] = 1

    adj = np.zeros((N, N))
    for j in m.GetBonds():
        u = j.GetBeginAtomIdx()
        v = j.GetEndAtomIdx()

        adj[u, v] = adj[v, u] = 1

    adj += np.eye(N)
    return nodes, adj


def prepare_data():
    raw_data = pd.read_csv('bace_docked.csv').drop('Unnamed: 0', axis=1)

    data = {}
    for i, g in raw_data.groupby('scaffold_split'):
        data[i] = [(smiles2graph(v['smiles']), v) for v in g.to_dict('record')]

    return data['train'], data['test'], data['valid']

def pad_data(data):
    most_atoms = max(data, key=lambda x: x[0][0].shape[0])[0][0].shape[0]

    end_data = []
    for (nodes, adj_mat), label in data:
        size = nodes.shape[0]
        nodes = np.lib.pad(nodes, ((0, most_atoms - size), (0, 0)), constant_values=0)
        adj_mat = np.lib.pad(adj_mat, (0, most_atoms - size), constant_values=0)
        end_data.append(((nodes, adj_mat), label))

    return end_data


class MolDataset(torch.utils.data.Dataset):
    def __init__(self, raw_data, transform=None, target_transform=None):
        self.raw_data = raw_data

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        arg, label = self.raw_data[idx]

        if self.transform:
            arg = self.transform(arg)
        if self.target_transform:
            label = self.target_transform(label)

        return arg, label
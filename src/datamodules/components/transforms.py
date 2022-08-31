import networkx as nx
import torch
from torch_geometric.datasets import TUDataset  # , Planetoid
from torch_geometric.transforms import Compose
from torch_geometric.utils import from_networkx, to_networkx


class Identity:
    def __init__(self, in_degree=False, cat=False):
        self.in_degree = in_degree
        self.cat = cat

    def __call__(self, data):
        data.x = torch.ones(data.num_nodes).view(-1, 1)
        # data.x = torch.eye(data.num_nodes)
        return data


class NetworkXTransform:
    def __init__(self, cat=False):
        self.cat = cat

    def __call__(self, data):
        x = data.x
        netx_data = to_networkx(data).to_undirected()
        ecc = self.nx_transform(netx_data)
        nx.set_node_attributes(netx_data, ecc, "x")
        ret_data = from_networkx(netx_data)
        ret_x = ret_data.x.view(-1, 1).type(torch.float32)
        if x is not None and self.cat:
            x = x.view(-1, 1) if x.dim() == 1 else x
            data.x = torch.cat([x, ret_x], dim=-1)
        else:
            data.x = ret_x
        return data

    def nx_transform(self, networkx_data):
        """returns a node dictionary with a single attribute."""
        raise NotImplementedError


def _eccentricity_subgraphs(data):
    try:
        return nx.eccentricity(data)
    except nx.NetworkXError:
        # TODO make this better?
        return {i: 0 for i in range(len(data))}


class Eccentricity(NetworkXTransform):
    def nx_transform(self, data):
        return _eccentricity_subgraphs(data)


class ClusteringCoefficient(NetworkXTransform):
    def nx_transform(self, data):
        return nx.clustering(data)


class Standardize:
    def __init__(self, with_mean=True, with_std=True, cat=False):
        self.cat = cat
        self.with_mean = with_mean
        self.with_std = with_std

    def __call__(self, data):
        x = data.x
        y = x
        if self.with_mean:
            y = y - x.mean(dim=0)
        if self.with_std:
            y = y / x.std(dim=0) + 1e-8
        if x is not None and self.cat:
            x = x.view(-1, 1) if x.dim() == 1 else x
            data.x = torch.cat([x, y.to(x.device)], dim=-1)
        else:
            data.x = y
        return data


if __name__ == "__main__":
    transforms = Compose([Eccentricity(), ClusteringCoefficient(cat=True)])
    dataset = TUDataset(root="/tmp/tu", name="IMDB-BINARY", transform=transforms)

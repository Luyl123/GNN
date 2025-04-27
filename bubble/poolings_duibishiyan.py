from typing import Callable, Optional, Union

import torch
from torch.nn import Parameter
from torch_scatter import scatter, scatter_add, scatter_max, scatter_mean
from torch_geometric.data import Batch

from torch_geometric.utils import softmax, remove_self_loops, coalesce

from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.nn.inits import uniform


from torch_geometric.nn.pool.consecutive import consecutive_cluster
def model_1_topk(x, ratio, batch, min_score=None, tol=1e-7):
    if min_score is not None:
        # Make sure that we do not drop all nodes in a graph.
        scores_max = scatter_max(x, batch)[0].index_select(0, batch) - tol
        scores_min = scores_max.clamp(max=min_score)

        perm = (x > scores_min).nonzero(as_tuple=False).view(-1)
    else:
        num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
        batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()

        cum_num_nodes = torch.cat(
            [num_nodes.new_zeros(1),
             num_nodes.cumsum(dim=0)[:-1]], dim=0)

        index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
        index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)

        dense_x = x.new_full((batch_size * max_num_nodes, ),
                             torch.finfo(x.dtype).min)
        dense_x[index] = x
        dense_x = dense_x.view(batch_size, max_num_nodes)

        _, perm = dense_x.sort(dim=-1, descending=True)

        perm = perm + cum_num_nodes.view(-1, 1)
        perm = perm.view(-1)

        if isinstance(ratio, int):
            k = num_nodes.new_full((num_nodes.size(0), ), ratio)
            k = torch.min(k, num_nodes)
        else:
            k = (ratio * num_nodes.to(torch.float)).ceil().to(torch.long)

        mask = [
            torch.arange(k[i], dtype=torch.long, device=x.device) +
            i * max_num_nodes for i in range(batch_size)
        ]
        mask = torch.cat(mask, dim=0)

        perm = perm[mask]

    return perm


def filter_adj(edge_index, edge_attr, perm, num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    mask = perm.new_full((num_nodes, ), -1)
    i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
    mask[perm] = i

    row, col = edge_index
    row, col = mask[row], mask[col]
    mask = (row >= 0) & (col >= 0)
    row, col = row[mask], col[mask]

    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    return torch.stack([row, col], dim=0), edge_attr, mask


class model_1_TopKPooling_Mod(torch.nn.Module):
    def __init__(self, in_channels: int, ratio: Union[int, float] = 0.5,
                 min_score: Optional[float] = None, multiplier: float = 1.,
                 nonlinearity: Callable = torch.tanh):
        super().__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.min_score = min_score
        self.multiplier = multiplier
        self.nonlinearity = nonlinearity

        self.weight = Parameter(torch.Tensor(1, in_channels))

        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_channels
        uniform(size, self.weight)

    def forward(self, x, edge_index, perm,j,per_t,edge_attr=None, batch=None, attn=None):
        """"""

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        
        attn = x if attn is None else attn
        attn = attn.unsqueeze(-1) if attn.dim() == 1 else attn
        score = (attn * self.weight).sum(dim=-1)

        if self.min_score is None:
            score = self.nonlinearity(score / self.weight.norm(p=2, dim=-1))
        else:
            score = softmax(score, batch)
        if (j%per_t)==0:
          perm = model_1_topk(score, self.ratio, batch, self.min_score)
        x = x[perm] * score[perm].view(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x

        batch = batch[perm]
        edge_index, edge_attr, edge_mask = filter_adj(edge_index, edge_attr, perm,
                                           num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm, edge_mask, score[perm]

    def __repr__(self) -> str:
        if self.min_score is None:
            ratio = f'ratio={self.ratio}'
        else:
            ratio = f'min_score={self.min_score}'

        return (f'{self.__class__.__name__}({self.in_channels}, {ratio}, '
                f'multiplier={self.multiplier})')

# Edge pooling
def pool_edge_mean(cluster, edge_index, edge_attr: Optional[torch.Tensor] = None):
    num_nodes = cluster.size(0)
    edge_index = cluster[edge_index.view(-1)].view(2, -1) 
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    if edge_index.numel() > 0:
        edge_index, edge_attr = coalesce(edge_index, edge_attr, reduce='mean')
    return edge_index, edge_attr




# avg pooling 
def avg_pool_mod(cluster, x, edge_index, edge_attr, batch, pos):
    cluster, perm = consecutive_cluster(cluster)

    # Pool node attributes 
    # x_pool = None if x is None else _avg_pool_x(cluster, x)
    x_pool = None if x is None else scatter(x, cluster, dim=0, dim_size=None, reduce='mean')

    # Pool edge attributes 
    edge_index_pool, edge_attr_pool = pool_edge_mean(cluster, edge_index, edge_attr)

    # Pool batch 
    batch_pool = None if batch is None else batch[perm]

    # Pool node positions 
    pos_pool = None if pos is None else scatter_mean(pos, cluster, dim=0)

    return x_pool, edge_index_pool, edge_attr_pool, batch_pool, pos_pool, cluster, perm




def avg_pool_mod_no_x(cluster, edge_index, edge_attr, batch, pos):
    cluster, perm = consecutive_cluster(cluster)

    # Pool edge attributes 
    edge_index_pool, edge_attr_pool = pool_edge_mean(cluster, edge_index, edge_attr)

    # Pool batch 
    batch_pool = None if batch is None else batch[perm]

    # Pool node positions 
    pos_pool = None if pos is None else scatter_mean(pos, cluster, dim=0)

    return edge_index_pool, edge_attr_pool, batch_pool, pos_pool, cluster, perm


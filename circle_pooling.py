from typing import Callable, Optional, Union

import torch
from torch.nn import Parameter
from torch_scatter import scatter, scatter_add, scatter_max, scatter_mean
from torch_geometric.data import Batch

from torch_geometric.utils import softmax, remove_self_loops, coalesce

from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.nn.inits import uniform
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F

from torch_geometric.nn.pool.consecutive import consecutive_cluster

def select_points(score,pos,x_min,x_max,y_min,y_max):
    rows=pos.shape[0]
    
    range_mask = (pos[torch.arange(rows),0] >= x_min) & (pos[torch.arange(rows) ,0]<= x_max) & \
             (pos[torch.arange(rows),1] >= y_min) & (pos[torch.arange(rows),1]<= y_max)
    indices = torch.nonzero(range_mask)
    indices = indices.squeeze()
    #print(indices)
    #print(indices.shape)
    for i in range(score.shape[0]):
        if i in indices:
            score[i]=score[i]
        else :
            score[i]=torch.finfo(score.dtype).min
    return score

def topk(x, pos,ratio_1,ratio_2,ratio_3,ratio_4,ratio_5,ratio_6,x_min_1,x_max_1,y_min_1,y_max_1,
                   x_min_2,x_max_2,y_min_2,y_max_2,x_min_3,x_max_3,y_min_3,y_max_3,
                   x_min_4,x_max_4,y_min_4,y_max_4,
                   x_min_5,x_max_5,y_min_5,y_max_5,
                   x_min_6,x_max_6,y_min_6,y_max_6,
                   
                   
         batch, min_score=None, tol=1e-7):
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
        index = (index - cum_num_nodes[batch].to(x.device)) + (batch * max_num_nodes).to(x.device)

        dense_x = x.new_full((batch_size * max_num_nodes, ),
                             torch.finfo(x.dtype).min)
        dense_x[index] = x
        dense_x = dense_x.view(batch_size, max_num_nodes)


        mask_1=select_point_score(dense_x ,pos,x_min_1,x_max_1,y_min_1,y_max_1,cum_num_nodes,
                           x,ratio_1,num_nodes,max_num_nodes,batch_size)
        
        mask_2=select_point_score(dense_x ,pos,x_min_2,x_max_2,y_min_2,y_max_2,cum_num_nodes,
                           x,ratio_2,num_nodes,max_num_nodes,batch_size)
        mask_3=select_point_score(dense_x ,pos,x_min_3,x_max_3,y_min_3,y_max_3,cum_num_nodes,
                           x,ratio_3,num_nodes,max_num_nodes,batch_size)
        mask_4=select_point_score(dense_x ,pos,x_min_4,x_max_4,y_min_4,y_max_4,cum_num_nodes,
                           x,ratio_4,num_nodes,max_num_nodes,batch_size)
        mask_5=select_point_score(dense_x ,pos,x_min_5,x_max_5,y_min_5,y_max_5,cum_num_nodes,
                           x,ratio_5,num_nodes,max_num_nodes,batch_size)
        mask_6=select_point_score(dense_x ,pos,x_min_6,x_max_6,y_min_6,y_max_6,cum_num_nodes,
                           x,ratio_6,num_nodes,max_num_nodes,batch_size)
                          

        
        mask_=mask_1+mask_2+mask_3+mask_4+mask_5+mask_6
        

    return mask_
def select_point_score(dense_x ,pos,x_min_6,x_max_6,y_min_6,y_max_6,cum_num_nodes,x,ratio_6,num_nodes,max_num_nodes,batch_size):
    dense_x_6=select_points(dense_x ,pos,x_min_6,x_max_6,y_min_6,y_max_6)
    _, perm_6 = dense_x_6.sort(dim=-1, descending=True)

    perm_6 = perm_6 + cum_num_nodes.view(-1, 1).to(x.device)
    perm_6= perm_6.view(-1)

    if isinstance(ratio_6, int):
            k = num_nodes.new_full((num_nodes.size(0), ), ratio_6)
            k = torch.min(k, num_nodes)
    else:
            k = (ratio_6 * num_nodes.to(torch.float)).ceil().to(torch.long)

    mask = [
            torch.arange(k[i], dtype=torch.long, device=x.device) +
            i * max_num_nodes for i in range(batch_size)
        ]
    mask = torch.cat(mask, dim=0)

    perm_6 = perm_6[mask]
    mask_6=x.new_zeros(x.size(0))
    mask_6[perm_6]=1
    return mask_6

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


def TAPooling_Mod( x, pos,ratio_1,ratio_2,ratio_3,ratio_4,ratio_5,ratio_6,x_min_1,x_max_1,y_min_1,y_max_1,
                   x_min_2,x_max_2,y_min_2,y_max_2,x_min_3,x_max_3,y_min_3,y_max_3,
                   x_min_4,x_max_4,y_min_4,y_max_4,
                   x_min_5,x_max_5,y_min_5,y_max_5,
                    x_min_6,x_max_6,y_min_6,y_max_6,
                   
                   
                   
                   min_score,edge_index,edge_attr=None, batch=None, attn=None): 
        
        attn = x if attn is None else attn
        attn = attn.unsqueeze(-1) if attn.dim() == 1 else attn
        
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        lam=0.2
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        adj = sp.coo_matrix((torch.ones(edge_index.shape[1]).to('cpu'), (edge_index[0, :].to('cpu'), edge_index[1, :].to('cpu'))), 
                                    shape=(attn.shape[0], attn.shape[0]), dtype=np.float32)
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = adj + sp.eye(adj.shape[0])
        adj = torch.FloatTensor(np.array(adj.todense())).to(device)
        
        
        R=torch.matmul(attn,attn.T)
        #print(R.shape)    #N*N
        d=torch.sum(adj,dim=1)
        
        #print(d)
        D = torch.diag(d,0).to('cpu')
        #print(D.device)
        D_inv=torch.inverse(D)
        R_=R.to(device)*torch.matmul(D_inv.to(device),adj).to(device)
        y_local=F.softmax((torch.matmul(R_,torch.ones(R.shape[0]).reshape(-1,1).to(device))/R.shape[0]) ,dim=0) #局部投票(N,1)

        H=torch.matmul(torch.matmul(D_inv.to(device),adj),attn.to(device)).to(device)
        y_global=F.softmax(torch.matmul(H,torch.ones(H.shape[1]).reshape(-1,1).to(device))/H.shape[1],dim=0).to(device) #(N,1)
        
        score=y_local+y_global+lam*(d/x.shape[0]).reshape(x.shape[0],1)
        score=score.squeeze()

        #print(score.device)

        

        mask = topk(score, pos,ratio_1,ratio_2,ratio_3,ratio_4,ratio_5,ratio_6,x_min_1,x_max_1,y_min_1,y_max_1,
                   x_min_2,x_max_2,y_min_2,y_max_2,x_min_3,x_max_3,y_min_3,y_max_3,
                   x_min_4,x_max_4,y_min_4,y_max_4,
                   x_min_5,x_max_5,y_min_5,y_max_5,
                   x_min_6,x_max_6,y_min_6,y_max_6,
                  
                   
                     batch, min_score)
        #print(x.size(0))
        '''x = x[perm] #* score[perm].view(-1, 1)
        
        #print(x.shape)
        #x = self.multiplier * x if self.multiplier != 1 else x

        batch = batch[perm]
        #print(score.size(0))
        edge_index, edge_attr, edge_mask = filter_adj(edge_index, edge_attr, perm,
                                           num_nodes=score.size(0))'''

        return mask#x, edge_index, perm        #, edge_attr, batch, perm, edge_mask, score[perm]



class TopKPooling_Mod(torch.nn.Module):
    r""":math:`\mathrm{top}_k` pooling operator from the `"Graph U-Nets"
    <https://arxiv.org/abs/1905.05178>`_, `"Towards Sparse
    Hierarchical Graph Classifiers" <https://arxiv.org/abs/1811.01287>`_
    and `"Understanding Attention and Generalization in Graph Neural
    Networks" <https://arxiv.org/abs/1905.02850>`_ papers

    if min_score :math:`\tilde{\alpha}` is None:

        .. math::
            \mathbf{y} &= \frac{\mathbf{X}\mathbf{p}}{\| \mathbf{p} \|}

            \mathbf{i} &= \mathrm{top}_k(\mathbf{y})

            \mathbf{X}^{\prime} &= (\mathbf{X} \odot
            \mathrm{tanh}(\mathbf{y}))_{\mathbf{i}}

            \mathbf{A}^{\prime} &= \mathbf{A}_{\mathbf{i},\mathbf{i}}

    if min_score :math:`\tilde{\alpha}` is a value in [0, 1]:

        .. math::
            \mathbf{y} &= \mathrm{softmax}(\mathbf{X}\mathbf{p})

            \mathbf{i} &= \mathbf{y}_i > \tilde{\alpha}

            \mathbf{X}^{\prime} &= (\mathbf{X} \odot \mathbf{y})_{\mathbf{i}}

            \mathbf{A}^{\prime} &= \mathbf{A}_{\mathbf{i},\mathbf{i}},

    where nodes are dropped based on a learnable projection score
    :math:`\mathbf{p}`.

    Args:
        in_channels (int): Size of each input sample.
        ratio (float or int): Graph pooling ratio, which is used to compute
            :math:`k = \lceil \mathrm{ratio} \cdot N \rceil`, or the value
            of :math:`k` itself, depending on whether the type of :obj:`ratio`
            is :obj:`float` or :obj:`int`.
            This value is ignored if :obj:`min_score` is not :obj:`None`.
            (default: :obj:`0.5`)
        min_score (float, optional): Minimal node score :math:`\tilde{\alpha}`
            which is used to compute indices of pooled nodes
            :math:`\mathbf{i} = \mathbf{y}_i > \tilde{\alpha}`.
            When this value is not :obj:`None`, the :obj:`ratio` argument is
            ignored. (default: :obj:`None`)
        multiplier (float, optional): Coefficient by which features gets
            multiplied after pooling. This can be useful for large graphs and
            when :obj:`min_score` is used. (default: :obj:`1`)
        nonlinearity (torch.nn.functional, optional): The nonlinearity to use.
            (default: :obj:`torch.tanh`)
    """
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
        #lam=0.1
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        
        attn = x if attn is None else attn
        attn = attn.unsqueeze(-1) if attn.dim() == 1 else attn
        score = (attn * self.weight).sum(dim=-1)

        #device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #adj = sp.coo_matrix((torch.ones(edge_index.shape[1]).to('cpu'), (edge_index[0, :].to('cpu'), edge_index[1, :].to('cpu'))), 
                                    #shape=(attn.shape[0], attn.shape[0]), dtype=np.float32)
        # build symmetric adjacency matrix
        #adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        #adj = adj + sp.eye(adj.shape[0])
        #adj = torch.FloatTensor(np.array(adj.todense())).to(device)
        #d=torch.sum(adj,dim=1)

        if self.min_score is None:
            score = self.nonlinearity(score / self.weight.norm(p=2, dim=-1))
        else:
            score = softmax(score, batch)
        #score=score+lam*(d/x.shape[0])
        if (j%per_t)==0:
          perm = topk(score, self.ratio, batch, self.min_score)
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
    edge_index = cluster[edge_index.contiguous().view(-1)].view(2, -1) 
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
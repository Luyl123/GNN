import os, time
from typing import Optional, Union, Callable, List
import numpy as np
import torch
from torch_geometric.data import Data
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.inits import uniform
from torch_geometric.utils.repeat import repeat
from torch_geometric.utils import softmax
from torch_geometric.utils.to_dense_batch import to_dense_batch
import torch_geometric.nn as tgnn
from torch_geometric.nn.conv import MessagePassing
from tap_processing import mp_ae_dataprocess, mp_ae_dataprocess_inv
from torch_geometric.typing import (
    Adj,
    NoneType,
    OptPairTensor,
    OptTensor,
    Size,
)
from d_pooling import TAPooling_Mod, avg_pool_mod, avg_pool_mod_no_x,TopKPooling_Mod,filter_adj

class MessagePassing_Autoencode_attn(torch.nn.Module):
    def __init__(self,
                 hidden_channels: int, 
                 in_channels_node : int, 
                 in_channels_edge : int, 
                 out_channels : int, 
                 encoding_dim:int,
                 n_mlp_mp : int, 
                 n_mlp_encode:int,
                 ae_dim:List[int],
                 n_mp_down : List[int], 
                 n_mp_up : List[int], 
                 pool_num:List[int],
                 act : Optional[Callable] = F.elu, 
                 name : Optional[str] = 'mmp_layer',
                ):
        super().__init__()
        self.edge_aggregator = EdgeAggregation()
        self.hidden_channels = hidden_channels
        self.in_channels_node = in_channels_node
        self.in_channels_edge = in_channels_edge
        self.out_channels = out_channels
        self.act = act
        self.n_mlp_encode = n_mlp_encode    # number of MLP layers in node/edge encoding stage 
        self.n_mlp_decode = n_mlp_encode 
        
        self.n_mlp_mp = n_mlp_mp # number of MLP layers in node/edge update functions used in message passing blocks
        self.ae_dim=ae_dim
        self.n_mp_down = n_mp_down # number of message passing blocks in downsampling path 
        self.n_mp_up = n_mp_up # number of message passing blocks in upsampling path  
        
        self.depth = len(self.n_mp_up)-1 # depth of u net 
        self.pool_num=pool_num #[1000,500,200] 
        self.encoding_dim=encoding_dim

        self.name = name


        # ~~~~ Node encoder
        self.node_encode = torch.nn.ModuleList() 
        for i in range(self.n_mlp_encode): 
            if i == 0:
                input_features = in_channels_node 
                output_features = hidden_channels 
            else:
                input_features = hidden_channels 
                output_features = hidden_channels 
            self.node_encode.append( nn.Linear(input_features, output_features) )
        self.node_encode_norm = nn.LayerNorm(output_features)
       
        # ~~~~ Edge encoder 
        self.edge_encode = torch.nn.ModuleList() 
        for i in range(self.n_mlp_encode): 
            if i == 0:
                input_features = in_channels_edge
                output_features = hidden_channels 
            else:
                input_features = hidden_channels 
                output_features = hidden_channels 
            self.edge_encode.append( nn.Linear(input_features, output_features) )
        self.edge_encode_norm = nn.LayerNorm(output_features)

        # ~~~~ DOWNWARD Message Passing
        # Edge updates: 
        self.edge_down_mps = torch.nn.ModuleList() 
        self.edge_down_norms = torch.nn.ModuleList()

        # Loop through levels: 
        for m in range(len(n_mp_down)):
            n_mp = n_mp_down[m]
            edge_mp = torch.nn.ModuleList()
            edge_mp_norm = torch.nn.ModuleList()

            # Loop through message passing steps per level 
            for i in range(n_mp):
                temp = torch.nn.ModuleList()

                # Loop through layers in MLP
                for j in range(self.n_mlp_mp):
                    if j == 0:
                        input_features = hidden_channels*3
                        output_features = hidden_channels 
                    else:
                        input_features = hidden_channels
                        output_features = hidden_channels
                    temp.append( nn.Linear(input_features, output_features) )
                edge_mp.append(temp)
                edge_mp_norm.append( nn.LayerNorm(output_features) )

            self.edge_down_mps.append(edge_mp)
            self.edge_down_norms.append(edge_mp_norm)

        # Node updates: 
        self.node_down_mps = torch.nn.ModuleList()
        self.node_down_norms = torch.nn.ModuleList()

        # Loop through levels: 
        for m in range(len(n_mp_down)):
            n_mp = n_mp_down[m]
            node_mp = torch.nn.ModuleList()
            node_mp_norm = torch.nn.ModuleList()

            # Loop through message passing steps per level 
            for i in range(n_mp):
                temp = torch.nn.ModuleList()

            # Loop through layers in MLP
                for j in range(self.n_mlp_mp):
                        if m ==0:
                            if j==0:
                                input_features = hidden_channels*2
                                output_features = hidden_channels 
                            else:
                                input_features = hidden_channels
                                output_features = hidden_channels
                            temp.append( nn.Linear(input_features, output_features) )
                        else:
                            if j == 0:
                                input_features = hidden_channels*2+encoding_dim
                                output_features = hidden_channels 
                            else:
                                input_features = hidden_channels
                                output_features = hidden_channels
                            temp.append( nn.Linear(input_features, output_features) )
                node_mp.append(temp)
                node_mp_norm.append( nn.LayerNorm(output_features) )

            self.node_down_mps.append(node_mp)
            self.node_down_norms.append(node_mp_norm)

        # ~~~~ UPWARD Message Passing
        self.edge_up_mps = torch.nn.ModuleList() 
        self.edge_up_norms = torch.nn.ModuleList()

        # Loop through levels: 
        for m in range(len(n_mp_up)):
            n_mp = n_mp_up[m]
            edge_mp = torch.nn.ModuleList()
            edge_mp_norm = torch.nn.ModuleList()

            # Loop through message passing steps per level 
            for i in range(n_mp):
                temp = torch.nn.ModuleList()

                # Loop through layers in MLP
                for j in range(self.n_mlp_mp):
                    if j == 0:
                        input_features = hidden_channels*3
                        output_features = hidden_channels 
                    else:
                        input_features = hidden_channels
                        output_features = hidden_channels
                    temp.append( nn.Linear(input_features, output_features) )
                edge_mp.append(temp)
                edge_mp_norm.append( nn.LayerNorm(output_features) )

            self.edge_up_mps.append(edge_mp)
            self.edge_up_norms.append(edge_mp_norm)

        # Node updates: 
        self.node_up_mps = torch.nn.ModuleList()
        self.node_up_norms = torch.nn.ModuleList()

        # Loop through levels: 
        for m in range(len(n_mp_up)):
            n_mp = n_mp_up[m]
            node_mp = torch.nn.ModuleList()
            node_mp_norm = torch.nn.ModuleList()

            # Loop through message passing steps per level 
            for i in range(n_mp):
                temp = torch.nn.ModuleList()

                # Loop through layers in MLP
                for j in range(self.n_mlp_mp):
                    if m ==(len(n_mp_up)-1):
                        if j==0:
                            input_features = hidden_channels*2
                            output_features = hidden_channels 
                        else:
                            input_features = hidden_channels
                            output_features = hidden_channels
                        temp.append( nn.Linear(input_features, output_features) )
                    else:
                        if j == 0:
                            input_features = hidden_channels*2+encoding_dim
                            output_features = hidden_channels 
                        else:
                            input_features = hidden_channels
                            output_features = hidden_channels
                        temp.append( nn.Linear(input_features, output_features) )
                node_mp.append(temp)
                node_mp_norm.append( nn.LayerNorm(output_features))

            self.node_up_mps.append(node_mp)
            self.node_up_norms.append(node_mp_norm)

        # ~~~~ DOWNWARD autoencode
        self.autoencode_down = torch.nn.ModuleList()
        self.attn_score_down = torch.nn.ModuleList() 

        for m in range(len(n_mp_down)-1):
            n_ae=n_mp_down[m+1] 
            n_node_num=pool_num[m]
            node_ae = torch.nn.ModuleList()
            n_attn = torch.nn.ModuleList()

            for i in range(n_ae):
                temp = torch.nn.ModuleList()
                for j in range(len(self.ae_dim)):
                    if j==0:
                        input_features = hidden_channels*n_node_num
                        output_features = self.ae_dim[j]
                    else:
                        input_features = self.ae_dim[j-1]
                        output_features = self.ae_dim[j]

                    temp.append( nn.Linear(input_features, output_features) )
                node_ae.append(temp)
                n_attn.append(Global_attention(hidden_channels, encoding_dim))
            self.autoencode_down.append(node_ae)  
            self.attn_score_down.append(n_attn)  

        # ~~~~ UPWARD autoencode
        self.autoencode_up = torch.nn.ModuleList()
        self.attn_score_up = torch.nn.ModuleList()

        for m in range(len(n_mp_up)-1):
            n_ae=n_mp_up[m] 
            n_node_num=pool_num[self.depth-m-1]
            node_ae = torch.nn.ModuleList()
            n_attn = torch.nn.ModuleList()

            for i in range(n_ae):
                temp = torch.nn.ModuleList()
                for j in range(len(self.ae_dim)):
                    if j==0:
                        input_features = hidden_channels*n_node_num
                        output_features = self.ae_dim[j]
                    else:
                        input_features = self.ae_dim[j-1]
                        output_features = self.ae_dim[j]
                    temp.append( nn.Linear(input_features, output_features) )
                node_ae.append(temp)
                n_attn.append(Global_attention(hidden_channels, encoding_dim))
            self.autoencode_up.append(node_ae)
            self.attn_score_up.append(n_attn)


       
            


        # ~~~~ Node-wise decoder
        self.node_decode = torch.nn.ModuleList() 
        for i in range(self.n_mlp_decode): 
            if i == self.n_mlp_decode - 1:
                input_features = hidden_channels 
                output_features = out_channels
            else:
                input_features = hidden_channels 
                output_features = hidden_channels 
            self.node_decode.append( nn.Linear(input_features, output_features) )

        '''# ~~~~ POOLING  
        self.pools = torch.nn.ModuleList() # for pooling 
        for i in range(self.depth):
            self.pools.append(TopKPooling_Mod(hidden_channels, self.pool_num[i]))'''

            
        # Reset params 
        self.reset_parameters()

    def forward(self, data,per_t , perms,batch_size,mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge,batch=None, return_mask=True):
        
        
        
         
        
        if return_mask: 
            mask = data.x.new_zeros(data.x.size(0))
        else:
            mask = None
        
        pointnum=int(data.x.shape[0]/(per_t*batch_size))
        #edgenum=int(data.edge_attr.shape[0]/(per_t*batch_size))
        device = 'cuda' if torch.cuda.is_available() else 'cpu' 
        y=torch.tensor([[0,0]]).type(torch.float).to(device)
        #y=torch.tensor([[0]]).type(torch.float).to(device)
        for j in range(per_t*batch_size):
          batch=None
          x=data.x[j*pointnum:(j+1)*pointnum,:]
          edge_index=data.edge_index
          edge_attr=data.edge_attr
          pos=data.mesh_pos 
          #print(edge_index[0,:].shape)
          #print(edge_index[1,:].shape)
          #print(edge_attr .shape)
          
          x=(x-mean_vec_x)/std_vec_x
          edge_attr=(edge_attr-mean_vec_edge)/std_vec_edge
          if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        # ~~~~ Node Encoder: 
          for i in range(self.n_mlp_encode):
            x = self.node_encode[i](x) 
            if i < self.n_mlp_encode - 1:
                x = self.act(x)
            else:
                x = x
          x = self.node_encode_norm(x)
          
        # ~~~~ Edge Encoder: 
          for i in range(self.n_mlp_encode):
            edge_attr = self.edge_encode[i](edge_attr)
            if i < self.n_mlp_encode - 1:
                edge_attr = self.act(edge_attr)
            else:
                edge_attr = edge_attr
          edge_attr = self.edge_encode_norm(edge_attr)
          #print(edge_attr .shape)

          m = 0 # level index 
          n_mp = self.n_mp_down[m] # number of message passing blocks
        
          for i in range(n_mp):
              
              x_own = x[edge_index[0,:], :]
              x_nei = x[edge_index[1,:], :]
              edge_attr_t = torch.cat((x_own, x_nei, edge_attr), axis=1)
            
              # edge update mlp
              for g in range(self.n_mlp_mp):
                  edge_attr_t = self.edge_down_mps[m][i][g](edge_attr_t) 
                  if g < self.n_mlp_mp - 1:
                      edge_attr_t = self.act(edge_attr_t)

              edge_attr = edge_attr + edge_attr_t
              edge_attr = self.edge_down_norms[m][i](edge_attr)

              # edge aggregation 
              edge_agg = self.edge_aggregator(x, edge_index, edge_attr)

              
              #x_ae=mp_ae_dataprocess(x) 
              
              #for g in range(len(self.ae_dim)):
                  #x_ae = self.autoencode_down[m][i][g](x_ae)
                  
                  #if g <len(self.ae_dim) - 1:
                      #x_ae = self.act(x_ae)
              
              
              #不同节点对全局特征向量的重要有差异
              #g=self.attn_score_down[m][i](x, x_ae)
              #print(g.shape)
              #x_t = torch.cat((x, edge_agg,g), axis=1)
              x_t = torch.cat((x, edge_agg), axis=1)
              #print(x_t.shape)

              # node update mlp
              for g in range(self.n_mlp_mp):
                  x_t = self.node_down_mps[m][i][g](x_t)
                  if g < self.n_mlp_mp - 1:
                      x_t = self.act(x_t) 
              #print(x_t.shape)
              x = x + x_t
              x = self.node_down_norms[m][i](x)

          
          
          

          xs = [x] 
          positions = [pos]
          edge_indices = [edge_index]
          edge_attrs = [edge_attr]
          batches = [batch]
          
          edge_masks = []
          
          for m in range(1, self.depth + 1):
            # Pooling: returns new x and edge_index for coarser grid 
            perm=perms[m-1]
            edge_index, edge_attr, edge_mask = filter_adj(edge_index, edge_attr, perm,
                                           num_nodes=x.size(0))
            x = x[perm]
            batch = batch[perm]
           
            pos = pos[perm]

            # Append the permutation list for node upsampling
            

            # Append the edge mask list for edge upsampling
            edge_masks += [edge_mask]

            # Append the positions list for upsampling
            positions += [pos]

            # append the batch list for upsampling
            batches += [batch]
            
        
            for i in range(self.n_mp_down[m]):
                x_own = x[edge_index[0,:], :]
                x_nei = x[edge_index[1,:], :]
                edge_attr_t = torch.cat((x_own, x_nei, edge_attr), axis=1)
            
                # edge update mlp
                for g in range(self.n_mlp_mp):
                    edge_attr_t = self.edge_down_mps[m][i][g](edge_attr_t) 
                    if g < self.n_mlp_mp - 1:
                        edge_attr_t = self.act(edge_attr_t)

                edge_attr = edge_attr + edge_attr_t
                edge_attr = self.edge_down_norms[m][i](edge_attr)

                # edge aggregation 
                edge_agg = self.edge_aggregator(x, edge_index, edge_attr)

                x_ae=mp_ae_dataprocess(x) 
                #print(x.shape,x_ae.shape)
                for g in range(len(self.ae_dim)):
                    x_ae = self.autoencode_down[m-1][i][g](x_ae) 
                    if g <len(self.ae_dim) - 1:
                        x_ae = self.act(x_ae)
              
                #print(x.shape,x_ae.shape)
                #不同节点对全局特征向量的重要有差异
                g=self.attn_score_down[m-1][i](x, x_ae)


                x_t = torch.cat((x, edge_agg,g), axis=1)


                # node update mlp
                for g in range(self.n_mlp_mp):
                    x_t = self.node_down_mps[m][i][g](x_t)
                    if g < self.n_mlp_mp - 1:
                        x_t = self.act(x_t) 
            
                x = x + x_t
                x = self.node_down_norms[m][i](x)

            
            

            
            
            if m < self.depth:
                xs += [x]
                edge_indices += [edge_index]
                edge_attrs += [edge_attr]

        # ~~~~ Fill node mask:

          if return_mask: 
            print('Filling mask')
            perm_global = perms[0] 
            mask[perm_global] = 1
            for i in range(1,self.depth):
                perm_global = perm_global[perms[i]]
                mask[perm_global] = i+1 

        # ~~~~ Upward message passing (decoder)
          
          m = 0
         
          for i in range(self.n_mp_up[m]):
              x_own = x[edge_index[0,:], :]
              x_nei = x[edge_index[1,:], :]
              edge_attr_t = torch.cat((x_own, x_nei, edge_attr), axis=1)
            
            # edge update mlp
              for g in range(self.n_mlp_mp):
                  edge_attr_t = self.edge_up_mps[m][i][g](edge_attr_t) 
                  if g < self.n_mlp_mp - 1:
                      edge_attr_t = self.act(edge_attr_t)

              edge_attr = edge_attr + edge_attr_t
              edge_attr = self.edge_up_norms[m][i](edge_attr)

              # edge aggregation 
              edge_agg = self.edge_aggregator(x, edge_index, edge_attr)

              x_ae=mp_ae_dataprocess(x) 
              #print(x.shape,x_ae.shape)
              for g in range(len(self.ae_dim)):
                  x_ae = self.autoencode_up[m][i][g](x_ae) 
                  if g <len(self.ae_dim) - 1:
                      x_ae = self.act(x_ae)

              #print(x.shape,x_ae.shape)
              #不同节点对全局特征向量的重要有差异
              g=self.attn_score_up[m][i](x, x_ae)

              x_t = torch.cat((x, edge_agg,g), axis=1)

             

              # node update mlp
              for g in range(self.n_mlp_mp):
                  x_t = self.node_up_mps[m][i][g](x_t)
                  if g < self.n_mlp_mp - 1:
                      x_t = self.act(x_t) 
            
              x = x + x_t
              x = self.node_up_norms[m][i](x)

          
        # upward cycle
          for m in range(self.depth):
            # Get the fine level index
            fine = self.depth - 1 - m

            # Get the batch
            batch = batches[fine]

            # Get node features and edge features on fine level
            res = xs[fine]
            pos = positions[fine]
            res_edge = edge_attrs[fine]

            # Get edge index on fine level
            edge_index = edge_indices[fine]

            # Upsample edge features
            edge_mask = edge_masks[fine]
            up_edge = torch.zeros_like(res_edge)
            up_edge[edge_mask] = edge_attr
            edge_attr = up_edge

            # Upsample node features
            # get node assignments on fine level
            perm = perms[fine]
            up = torch.zeros_like(res)
            up[perm] = x
            x = up
            

            # Message passing on new upsampled graph
            for i in range(self.n_mp_up[m+1]):
                x_own = x[edge_index[0,:], :]
                x_nei = x[edge_index[1,:], :]
                edge_attr_t = torch.cat((x_own, x_nei, edge_attr), axis=1)
            
            # edge update mlp
                for g in range(self.n_mlp_mp):
                    edge_attr_t = self.edge_up_mps[m+1][i][g](edge_attr_t) 
                    if g < self.n_mlp_mp - 1:
                        edge_attr_t = self.act(edge_attr_t)

                edge_attr = edge_attr + edge_attr_t
                edge_attr = self.edge_up_norms[m+1][i](edge_attr)

              # edge aggregation 
                edge_agg = self.edge_aggregator(x, edge_index, edge_attr)

                if m< (self.depth-1) :
                    x_ae=mp_ae_dataprocess(x) 
                    #print(x.shape,x_ae.shape)
                    for g in range(len(self.ae_dim)):
                        x_ae = self.autoencode_up[m+1][i][g](x_ae) 
                        if g <len(self.ae_dim) - 1:
                            x_ae = self.act(x_ae)

              
                    #print(x.shape,x_ae.shape)
                    #不同节点对全局特征向量的重要有差异
                    g=self.attn_score_up[m+1][i](x, x_ae)

                    x_t = torch.cat((x, edge_agg,g), axis=1)
                else:
                    x_t = torch.cat((x, edge_agg), axis=1)


              # node update mlp
                for g in range(self.n_mlp_mp):
                    x_t = self.node_up_mps[m+1][i][g](x_t)
                    if g < self.n_mlp_mp - 1:
                        x_t = self.act(x_t) 
            
                x = x + x_t
                x = self.node_up_norms[m+1][i](x)


        # ~~~~ Node decoder
          
          for i in range(self.n_mlp_decode):
            x = self.node_decode[i](x) 
            if i < self.n_mlp_decode - 1:
                x = self.act(x)
            else:
                x = x
          y=torch.vstack((y,x))
        y=y[1:,:]
        return y, mask 
    def input_dict(self):
        a = { 'hidden_channels' : self.hidden_channels, 
              'in_channels_node' : self.in_channels_node, 
              'in_channels_edge' : self.in_channels_edge,
               'out_channels' : self.out_channels, 
                'n_mlp_mp' : self.n_mlp_mp, 
                'n_mlp_encode' : self.n_mlp_encode,
                'ae_dim':self.ae_dim,
                'n_mp_down' : self.n_mp_down, 
                'n_mp_up' : self.n_mp_up, 
                'pool_num' : self.pool_num, 
                
                'encoding_dim':self.encoding_dim,
                
                
                'depth' : self.depth, 
                'act' : self.act, 
                
                'name' : self.name }

        return a

    def reset_parameters(self):
        # Node encoding
        for module in self.node_encode:
            module.reset_parameters()

        # Edge encoding 
        for module in self.edge_encode:
            module.reset_parameters()

        # Node decoder: 
        for module in self.node_decode:
            module.reset_parameters()

        

        for modulelist_level in self.attn_score_down:
            for module in modulelist_level:
                module.reset_parameters()

        for modulelist_level in self.attn_score_up:
            for module in modulelist_level:
                module.reset_parameters()

        
        for modulelist_level in self.edge_down_mps:
            for modulelist_mp in modulelist_level:
                for module in modulelist_mp: 
                    module.reset_parameters()

        # Down Message passing, node update 
        for modulelist_level in self.node_down_mps:
            for modulelist_mp in modulelist_level:
                for module in modulelist_mp: 
                    module.reset_parameters()

        # Up message passing, edge update 
        for modulelist_level in self.edge_up_mps:
            for modulelist_mp in modulelist_level:
                for module in modulelist_mp: 
                    module.reset_parameters()

        # Up message passing, node update 
        for modulelist_level in self.node_up_mps:
            for modulelist_mp in modulelist_level:
                for module in modulelist_mp: 
                    module.reset_parameters() 

        for modulelist_level in self.autoencode_up:
            for modulelist_mp in modulelist_level:
                for module in modulelist_mp: 
                    
                    module.reset_parameters() 
                    
        for modulelist_level in self.autoencode_down:
            for modulelist_mp in modulelist_level:
                for module in modulelist_mp:
                  module.reset_parameters() 

class EdgeAggregation(MessagePassing):
    def __init__(self, **kwargs):
        kwargs.setdefault('aggr', 'mean')
        super().__init__(**kwargs)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        x_j = edge_attr
        return x_j

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'
    




class Global_attention(torch.nn.Module):  
    def __init__(self, hidden_channels: int,encoding_dim: int):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.encoding_dim=encoding_dim

        self.weight = Parameter(torch.Tensor(1, hidden_channels+encoding_dim))

        self.reset_parameters()

    def reset_parameters(self):
        size = self.hidden_channels+self.encoding_dim
        uniform(size, self.weight)

    def forward(self, x, g_ae, attn=None):
        """"""
        attn = x if attn is None else attn
        attn = attn.unsqueeze(-1) if attn.dim() == 1 else attn
        
        g_=g_ae.repeat(attn.shape[0],1)
        #print(g_.shape) #(点数，encoding_dim)   
        attn_=torch.cat((attn,g_),dim=-1)
        
        score = (attn_ * self.weight).sum(dim=-1)
        score=score.reshape(1,-1)
        
        score=F.softmax(score,dim=1)
        
        
        g_1=(g_.T*score)
        gg=g_1.T
        #print(gg.shape)#(点数，encoding_dim)
        return gg

    def __repr__(self) -> str:
        if self.min_score is None:
            ratio = f'ratio={self.ratio}'
        else:
            ratio = f'min_score={self.min_score}'

        return (f'{self.__class__.__name__}({self.in_channels}, {ratio}, '
                f'multiplier={self.multiplier})')







class GNN_topk_encode(torch.nn.Module):
    def __init__(self,
                 hidden_channels: int, 
                 in_channels_node : int, 
                 in_channels_edge : int, 
                 out_channels : int, 
                 encoding_dim:int,
                 n_mlp_mp : int, 
                 n_mlp_encode:int,
                 ae_dim:List[int],
                 n_mp_down : List[int], 
                 n_mp_up : List[int], 
                 pool_num:List[int],
                 act : Optional[Callable] = F.elu, 
                 name : Optional[str] = 'mmp_layer',
                ):
        super().__init__()
        self.edge_aggregator = EdgeAggregation()
        self.hidden_channels = hidden_channels
        self.in_channels_node = in_channels_node
        self.in_channels_edge = in_channels_edge
        self.out_channels = out_channels
        self.act = act
        self.n_mlp_encode = n_mlp_encode    # number of MLP layers in node/edge encoding stage 
        self.n_mlp_decode = n_mlp_encode 
        
        self.n_mlp_mp = n_mlp_mp # number of MLP layers in node/edge update functions used in message passing blocks
        self.ae_dim=ae_dim
        self.n_mp_down = n_mp_down # number of message passing blocks in downsampling path 
        self.n_mp_up = n_mp_up # number of message passing blocks in upsampling path  
        
        self.depth = len(self.n_mp_up)-1 # depth of u net 
        self.pool_num=pool_num #[1000,500,200] 
        self.encoding_dim=encoding_dim

        self.name = name

        # ~~~~ Node encoder
        self.node_encode = torch.nn.ModuleList() 
        for i in range(self.n_mlp_encode): 
            if i == 0:
                input_features = in_channels_node 
                output_features = hidden_channels 
            else:
                input_features = hidden_channels 
                output_features = hidden_channels 
            self.node_encode.append( nn.Linear(input_features, output_features) )
        self.node_encode_norm = nn.LayerNorm(output_features)
       
        # ~~~~ Edge encoder 
        self.edge_encode = torch.nn.ModuleList() 
        for i in range(self.n_mlp_encode): 
            if i == 0:
                input_features = in_channels_edge
                output_features = hidden_channels 
            else:
                input_features = hidden_channels 
                output_features = hidden_channels 
            self.edge_encode.append( nn.Linear(input_features, output_features) )
        self.edge_encode_norm = nn.LayerNorm(output_features)

        # ~~~~ DOWNWARD Message Passing
        # Edge updates: 
        self.edge_down_mps = torch.nn.ModuleList() 
        self.edge_down_norms = torch.nn.ModuleList()

        # Loop through levels: 
        for m in range(len(n_mp_down)):
            n_mp = n_mp_down[m]
            edge_mp = torch.nn.ModuleList()
            edge_mp_norm = torch.nn.ModuleList()

            # Loop through message passing steps per level 
            for i in range(n_mp):
                temp = torch.nn.ModuleList()

                # Loop through layers in MLP
                for j in range(self.n_mlp_mp):
                    if j == 0:
                        input_features = hidden_channels*3
                        output_features = hidden_channels 
                    else:
                        input_features = hidden_channels
                        output_features = hidden_channels
                    temp.append( nn.Linear(input_features, output_features) )
                edge_mp.append(temp)
                edge_mp_norm.append( nn.LayerNorm(output_features) )

            self.edge_down_mps.append(edge_mp)
            self.edge_down_norms.append(edge_mp_norm)

        # Node updates: 
        self.node_down_mps = torch.nn.ModuleList()
        self.node_down_norms = torch.nn.ModuleList()

        # Loop through levels: 
        for m in range(len(n_mp_down)):
            n_mp = n_mp_down[m]
            node_mp = torch.nn.ModuleList()
            node_mp_norm = torch.nn.ModuleList()

            # Loop through message passing steps per level 
            for i in range(n_mp):
                temp = torch.nn.ModuleList()

            # Loop through layers in MLP
                for j in range(self.n_mlp_mp):
                        if m ==0:
                            if j==0:
                                input_features = hidden_channels*2
                                output_features = hidden_channels 
                            else:
                                input_features = hidden_channels
                                output_features = hidden_channels
                            temp.append( nn.Linear(input_features, output_features) )
                        else:
                            if j == 0:
                                input_features = hidden_channels*2+encoding_dim
                                output_features = hidden_channels 
                            else:
                                input_features = hidden_channels
                                output_features = hidden_channels
                            temp.append( nn.Linear(input_features, output_features) )
                node_mp.append(temp)
                node_mp_norm.append( nn.LayerNorm(output_features) )

            self.node_down_mps.append(node_mp)
            self.node_down_norms.append(node_mp_norm)

        self.autoencode_down = torch.nn.ModuleList()
        self.attn_score_down = torch.nn.ModuleList() 

        for m in range(len(n_mp_down)-1):
            n_ae=n_mp_down[m+1] 
            n_node_num=pool_num[m]
            node_ae = torch.nn.ModuleList()
            n_attn = torch.nn.ModuleList()

            for i in range(n_ae):
                temp = torch.nn.ModuleList()
                for j in range(len(self.ae_dim)):
                    if j==0:
                        input_features = hidden_channels*n_node_num
                        output_features = self.ae_dim[j]
                    else:
                        input_features = self.ae_dim[j-1]
                        output_features = self.ae_dim[j]

                    temp.append( nn.Linear(input_features, output_features) )
                node_ae.append(temp)
                n_attn.append(Global_attention(hidden_channels, encoding_dim))
            self.autoencode_down.append(node_ae)  
            self.attn_score_down.append(n_attn)
        
        
        

        # ~~~~ Reset params upon initialization
        self.reset_parameters()

    def forward(self, data,per_t,perms , batch_size,mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge,batch=None, return_mask=True):
        
      #batch_size为总的数据量/per_t
        
        
        
        
        if return_mask: 
            mask = data.x.new_zeros(data.x.size(0))
        else:
            mask = None
        
        pointnum=int(data.x.shape[0]/(per_t*batch_size))
        
        
        data_list=[]
        
        for j in range(per_t*batch_size):
          batch=None
          x=data.x[j*pointnum:(j+1)*pointnum,:]
          edge_index=data.edge_index
          edge_attr=data.edge_attr
          pos=data.mesh_pos
          #print(edge_index[0,:].shape)
          #print(edge_index[1,:].shape)
          #print(edge_attr .shape)
          
          x=(x-mean_vec_x)/std_vec_x
          edge_attr=(edge_attr-mean_vec_edge)/std_vec_edge
          if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        # ~~~~ Node Encoder: 
          for i in range(self.n_mlp_encode):
            x = self.node_encode[i](x) 
            if i < self.n_mlp_encode - 1:
                x = self.act(x)
            else:
                x = x
          x = self.node_encode_norm(x)
          
        # ~~~~ Edge Encoder: 
          for i in range(self.n_mlp_encode):
            edge_attr = self.edge_encode[i](edge_attr)
            if i < self.n_mlp_encode - 1:
                edge_attr = self.act(edge_attr)
            else:
                edge_attr = edge_attr
          edge_attr = self.edge_encode_norm(edge_attr)
          #print(edge_attr .shape)

          m = 0 # level index 
          n_mp = self.n_mp_down[m] # number of message passing blocks
        
          for i in range(n_mp):
              
              x_own = x[edge_index[0,:], :]
              x_nei = x[edge_index[1,:], :]
              edge_attr_t = torch.cat((x_own, x_nei, edge_attr), axis=1)
            
              # edge update mlp
              for g in range(self.n_mlp_mp):
                  edge_attr_t = self.edge_down_mps[m][i][g](edge_attr_t) 
                  if g < self.n_mlp_mp - 1:
                      edge_attr_t = self.act(edge_attr_t)

              edge_attr = edge_attr + edge_attr_t
              edge_attr = self.edge_down_norms[m][i](edge_attr)

              # edge aggregation 
              edge_agg = self.edge_aggregator(x, edge_index, edge_attr)

              
              #x_ae=mp_ae_dataprocess(x) 
              
              #for g in range(len(self.ae_dim)):
                  #x_ae = self.autoencode_down[m][i][g](x_ae)
                  
                  #if g <len(self.ae_dim) - 1:
                      #x_ae = self.act(x_ae)
              
              
              #不同节点对全局特征向量的重要有差异
              #g=self.attn_score_down[m][i](x, x_ae)
              #print(g.shape)
              #x_t = torch.cat((x, edge_agg,g), axis=1)
              x_t = torch.cat((x, edge_agg), axis=1)
              #print(x_t.shape)

              # node update mlp
              for g in range(self.n_mlp_mp):
                  x_t = self.node_down_mps[m][i][g](x_t)
                  if g < self.n_mlp_mp - 1:
                      x_t = self.act(x_t) 
              #print(x_t.shape)
              x = x + x_t
              x = self.node_down_norms[m][i](x)

          
          
          

          xs = [x] 
          positions = [pos]
          edge_indices = [edge_index]
          edge_attrs = [edge_attr]
          batches = [batch]
          
          edge_masks = []
          

          for m in range(1, self.depth + 1):
            # Pooling: returns new x and edge_index for coarser grid 
            perm=perms[m-1]
            edge_index, edge_attr, edge_mask = filter_adj(edge_index, edge_attr, perm,
                                           num_nodes=x.size(0))
            x = x[perm]
            batch = batch[perm]
           
            pos = pos[perm]
            

            # Append the permutation list for node upsampling
            

            # Append the edge mask list for edge upsampling
            edge_masks += [edge_mask]

            # Append the positions list for upsampling
            positions += [pos]

            # append the batch list for upsampling
            batches += [batch]
            
        
            for i in range(self.n_mp_down[m]):
                x_own = x[edge_index[0,:], :]
                x_nei = x[edge_index[1,:], :]
                edge_attr_t = torch.cat((x_own, x_nei, edge_attr), axis=1)
            
                # edge update mlp
                for g in range(self.n_mlp_mp):
                    edge_attr_t = self.edge_down_mps[m][i][g](edge_attr_t) 
                    if g < self.n_mlp_mp - 1:
                        edge_attr_t = self.act(edge_attr_t)

                edge_attr = edge_attr + edge_attr_t
                edge_attr = self.edge_down_norms[m][i](edge_attr)

                # edge aggregation 
                edge_agg = self.edge_aggregator(x, edge_index, edge_attr)

                x_ae=mp_ae_dataprocess(x) 
                #print(x.shape,x_ae.shape)
                for g in range(len(self.ae_dim)):
                    x_ae = self.autoencode_down[m-1][i][g](x_ae) 
                    if g <len(self.ae_dim) - 1:
                        x_ae = self.act(x_ae)
              
                #print(x.shape,x_ae.shape)
                #不同节点对全局特征向量的重要有差异
                g=self.attn_score_down[m-1][i](x, x_ae)


                x_t = torch.cat((x, edge_agg,g), axis=1)


                # node update mlp
                for g in range(self.n_mlp_mp):
                    x_t = self.node_down_mps[m][i][g](x_t)
                    if g < self.n_mlp_mp - 1:
                        x_t = self.act(x_t) 
            
                x = x + x_t
                x = self.node_down_norms[m][i](x)

            if m < self.depth:
                xs += [x]
                edge_indices += [edge_index]
                edge_attrs += [edge_attr]

        # ~~~~ Fill node mask:

          if return_mask: 
            print('Filling mask')
            perm_global = perms[0] 
            mask[perm_global] = 1
            for i in range(1,self.depth):
                perm_global = perm_global[perms[i]]
                mask[perm_global] = i+1
          data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))
        return data_list,batches , xs, positions, edge_attrs, edge_indices, edge_masks
    def input_dict(self):
        a = { 'hidden_channels' : self.hidden_channels, 
              'in_channels_node' : self.in_channels_node, 
              'in_channels_edge' : self.in_channels_edge,
               'out_channels' : self.out_channels, 
                'n_mlp_mp' : self.n_mlp_mp, 
                'n_mlp_encode' : self.n_mlp_encode,
                'ae_dim':self.ae_dim,
                'n_mp_down' : self.n_mp_down, 
                'n_mp_up' : self.n_mp_up, 
                'pool_num' : self.pool_num, 
                
                'encoding_dim':self.encoding_dim,
                
                
                'depth' : self.depth, 
                'act' : self.act, 
                
                'name' : self.name }

        return a

    def reset_parameters(self):
        # Node encoding
        for module in self.node_encode:
            module.reset_parameters()

        # Edge encoding 
        for module in self.edge_encode:
            module.reset_parameters()

        

       

        for modulelist_level in self.attn_score_down:
            for module in modulelist_level:
                module.reset_parameters()

    
        
        for modulelist_level in self.edge_down_mps:
            for modulelist_mp in modulelist_level:
                for module in modulelist_mp: 
                    module.reset_parameters()

        # Down Message passing, node update 
        for modulelist_level in self.node_down_mps:
            for modulelist_mp in modulelist_level:
                for module in modulelist_mp: 
                    module.reset_parameters()

                
        for modulelist_level in self.autoencode_down:
            for modulelist_mp in modulelist_level:
                for module in modulelist_mp:
                  module.reset_parameters() 







class GNN_encode(torch.nn.Module):
    def __init__(self,
                 hidden_channels: int, 
                 in_channels_node : int, 
                 in_channels_edge : int, 
                 out_channels : int, 
                 encoding_dim:int,
                 n_mlp_mp : int, 
                 n_mlp_encode:int,
                 ae_dim:List[int],
                 n_mp_down : List[int], 
                 n_mp_up : List[int], 
                 pool_num:List[int],
                 act : Optional[Callable] = F.elu, 
                 name : Optional[str] = 'mmp_layer',
                ):
        super().__init__()
        self.edge_aggregator = EdgeAggregation()
        self.hidden_channels = hidden_channels
        self.in_channels_node = in_channels_node
        self.in_channels_edge = in_channels_edge
        self.out_channels = out_channels
        self.act = act
        self.n_mlp_encode = n_mlp_encode    # number of MLP layers in node/edge encoding stage 
        self.n_mlp_decode = n_mlp_encode 
        
        self.n_mlp_mp = n_mlp_mp # number of MLP layers in node/edge update functions used in message passing blocks
        self.ae_dim=ae_dim
        self.n_mp_down = n_mp_down # number of message passing blocks in downsampling path 
        self.n_mp_up = n_mp_up # number of message passing blocks in upsampling path  
        
        self.depth = len(self.n_mp_up)-1 # depth of u net 
        self.pool_num=pool_num #[1000,500,200] 
        self.encoding_dim=encoding_dim

        self.name = name

        # ~~~~ Node encoder
        self.node_encode = torch.nn.ModuleList() 
        for i in range(self.n_mlp_encode): 
            if i == 0:
                input_features = in_channels_node 
                output_features = hidden_channels 
            else:
                input_features = hidden_channels 
                output_features = hidden_channels 
            self.node_encode.append( nn.Linear(input_features, output_features) )
        self.node_encode_norm = nn.LayerNorm(output_features)
       
        # ~~~~ Edge encoder 
        self.edge_encode = torch.nn.ModuleList() 
        for i in range(self.n_mlp_encode): 
            if i == 0:
                input_features = in_channels_edge
                output_features = hidden_channels 
            else:
                input_features = hidden_channels 
                output_features = hidden_channels 
            self.edge_encode.append( nn.Linear(input_features, output_features) )
        self.edge_encode_norm = nn.LayerNorm(output_features)

        # ~~~~ DOWNWARD Message Passing
        # Edge updates: 
        self.edge_down_mps = torch.nn.ModuleList() 
        self.edge_down_norms = torch.nn.ModuleList()

        # Loop through levels: 
        for m in range(len(n_mp_down)):
            n_mp = n_mp_down[m]
            edge_mp = torch.nn.ModuleList()
            edge_mp_norm = torch.nn.ModuleList()

            # Loop through message passing steps per level 
            for i in range(n_mp):
                temp = torch.nn.ModuleList()

                # Loop through layers in MLP
                for j in range(self.n_mlp_mp):
                    if j == 0:
                        input_features = hidden_channels*3
                        output_features = hidden_channels 
                    else:
                        input_features = hidden_channels
                        output_features = hidden_channels
                    temp.append( nn.Linear(input_features, output_features) )
                edge_mp.append(temp)
                edge_mp_norm.append( nn.LayerNorm(output_features) )

            self.edge_down_mps.append(edge_mp)
            self.edge_down_norms.append(edge_mp_norm)

        # Node updates: 
        self.node_down_mps = torch.nn.ModuleList()
        self.node_down_norms = torch.nn.ModuleList()

        # Loop through levels: 
        for m in range(len(n_mp_down)):
            n_mp = n_mp_down[m]
            node_mp = torch.nn.ModuleList()
            node_mp_norm = torch.nn.ModuleList()

            # Loop through message passing steps per level 
            for i in range(n_mp):
                temp = torch.nn.ModuleList()

            # Loop through layers in MLP
                for j in range(self.n_mlp_mp):
                        if m ==0:
                            if j==0:
                                input_features = hidden_channels*2
                                output_features = hidden_channels 
                            else:
                                input_features = hidden_channels
                                output_features = hidden_channels
                            temp.append( nn.Linear(input_features, output_features) )
                        else:
                            if j == 0:
                                input_features = hidden_channels*2+encoding_dim
                                output_features = hidden_channels 
                            else:
                                input_features = hidden_channels
                                output_features = hidden_channels
                            temp.append( nn.Linear(input_features, output_features) )
                node_mp.append(temp)
                node_mp_norm.append( nn.LayerNorm(output_features) )

            self.node_down_mps.append(node_mp)
            self.node_down_norms.append(node_mp_norm)

        self.autoencode_down = torch.nn.ModuleList()
        self.attn_score_down = torch.nn.ModuleList() 

        for m in range(len(n_mp_down)-1):
            n_ae=n_mp_down[m+1] 
            n_node_num=pool_num[m]
            node_ae = torch.nn.ModuleList()
            n_attn = torch.nn.ModuleList()

            for i in range(n_ae):
                temp = torch.nn.ModuleList()
                for j in range(len(self.ae_dim)):
                    if j==0:
                        input_features = hidden_channels*n_node_num
                        output_features = self.ae_dim[j]
                    else:
                        input_features = self.ae_dim[j-1]
                        output_features = self.ae_dim[j]

                    temp.append( nn.Linear(input_features, output_features) )
                node_ae.append(temp)
                n_attn.append(Global_attention(hidden_channels, encoding_dim))
            self.autoencode_down.append(node_ae)  
            self.attn_score_down.append(n_attn)
        
        # ~~~~ POOLING  
        self.pools = torch.nn.ModuleList() # for pooling 
        for i in range(self.depth):
            self.pools.append(TAPooling_Mod(hidden_channels, self.pool_num[i]))
        

        # ~~~~ Reset params upon initialization
        self.reset_parameters()

    def forward(self, data,per_t,mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge,batch=None, return_mask=True):
        
      #data一个时刻的的原图
        
        
        
       
        if return_mask: 
            mask = data.x.new_zeros(data.x.size(0))
        else:
            mask = None
        
        seq_len=1
        data_list=[]
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for j in range(seq_len):
          
          batch=None
          data.to(device)
          x=data.x
          edge_index=data.edge_index
          edge_attr=data.edge_attr
          pos=data.mesh_pos
          #print(edge_index[0,:].shape)
          #print(edge_index[1,:].shape)
          #print(edge_attr .shape)
          
          x=(x-mean_vec_x)/std_vec_x
          edge_attr=(edge_attr-mean_vec_edge)/std_vec_edge
          if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        # ~~~~ Node Encoder: 
          for i in range(self.n_mlp_encode):
            x = self.node_encode[i](x) 
            if i < self.n_mlp_encode - 1:
                x = self.act(x)
            else:
                x = x
          x = self.node_encode_norm(x)
          
        # ~~~~ Edge Encoder: 
          for i in range(self.n_mlp_encode):
            edge_attr = self.edge_encode[i](edge_attr)
            if i < self.n_mlp_encode - 1:
                edge_attr = self.act(edge_attr)
            else:
                edge_attr = edge_attr
          edge_attr = self.edge_encode_norm(edge_attr)
          #print(edge_attr .shape)

          m = 0 # level index 
          n_mp = self.n_mp_down[m] # number of message passing blocks
        
          for i in range(n_mp):
              
              x_own = x[edge_index[0,:], :]
              x_nei = x[edge_index[1,:], :]
              edge_attr_t = torch.cat((x_own, x_nei, edge_attr), axis=1)
            
              # edge update mlp
              for g in range(self.n_mlp_mp):
                  edge_attr_t = self.edge_down_mps[m][i][g](edge_attr_t) 
                  if g < self.n_mlp_mp - 1:
                      edge_attr_t = self.act(edge_attr_t)

              edge_attr = edge_attr + edge_attr_t
              edge_attr = self.edge_down_norms[m][i](edge_attr)

              # edge aggregation 
              edge_agg = self.edge_aggregator(x, edge_index, edge_attr)

              
              #x_ae=mp_ae_dataprocess(x) 
              
              #for g in range(len(self.ae_dim)):
                  #x_ae = self.autoencode_down[m][i][g](x_ae)
                  
                  #if g <len(self.ae_dim) - 1:
                      #x_ae = self.act(x_ae)
              
              
              #不同节点对全局特征向量的重要有差异
              #g=self.attn_score_down[m][i](x, x_ae)
              #print(g.shape)
              #x_t = torch.cat((x, edge_agg,g), axis=1)
              x_t = torch.cat((x, edge_agg), axis=1)
              #print(x_t.shape)

              # node update mlp
              for g in range(self.n_mlp_mp):
                  x_t = self.node_down_mps[m][i][g](x_t)
                  if g < self.n_mlp_mp - 1:
                      x_t = self.act(x_t) 
              #print(x_t.shape)
              x = x + x_t
              x = self.node_down_norms[m][i](x)

          
          
          

          xs = [x] 
          positions = [pos]
          edge_indices = [edge_index]
          edge_attrs = [edge_attr]
          batches = [batch]
          
          edge_masks = []
          if j==0:
              perms = []
              perms_=[]

          for m in range(1, self.depth + 1):
            # Pooling: returns new x and edge_index for coarser grid 
            if j==0:
              x, edge_index, edge_attr, batch, perm, edge_mask, _ = self.pools[m - 1](x,edge_index,perms_,j,per_t,edge_attr,batch)
              perms += [perm]
            if j>=1:
              x, edge_index, edge_attr, batch, perm, edge_mask, _ = self.pools[m - 1](x,edge_index,perms[m-1],j,per_t,edge_attr,batch)

            pos = pos[perm]

            # Append the permutation list for node upsampling
            

            # Append the edge mask list for edge upsampling
            edge_masks += [edge_mask]

            # Append the positions list for upsampling
            positions += [pos]

            # append the batch list for upsampling
            batches += [batch]
            
        
            for i in range(self.n_mp_down[m]):
                x_own = x[edge_index[0,:], :]
                x_nei = x[edge_index[1,:], :]
                edge_attr_t = torch.cat((x_own, x_nei, edge_attr), axis=1)
            
                # edge update mlp
                for g in range(self.n_mlp_mp):
                    edge_attr_t = self.edge_down_mps[m][i][g](edge_attr_t) 
                    if g < self.n_mlp_mp - 1:
                        edge_attr_t = self.act(edge_attr_t)

                edge_attr = edge_attr + edge_attr_t
                edge_attr = self.edge_down_norms[m][i](edge_attr)

                # edge aggregation 
                edge_agg = self.edge_aggregator(x, edge_index, edge_attr)

                x_ae=mp_ae_dataprocess(x) 
                #print(x.shape,x_ae.shape)
                for g in range(len(self.ae_dim)):
                    x_ae = self.autoencode_down[m-1][i][g](x_ae) 
                    if g <len(self.ae_dim) - 1:
                        x_ae = self.act(x_ae)
              
                #print(x.shape,x_ae.shape)
                #不同节点对全局特征向量的重要有差异
                g=self.attn_score_down[m-1][i](x, x_ae)


                x_t = torch.cat((x, edge_agg,g), axis=1)


                # node update mlp
                for g in range(self.n_mlp_mp):
                    x_t = self.node_down_mps[m][i][g](x_t)
                    if g < self.n_mlp_mp - 1:
                        x_t = self.act(x_t) 
            
                x = x + x_t
                x = self.node_down_norms[m][i](x)

            if m < self.depth:
                xs += [x]
                edge_indices += [edge_index]
                edge_attrs += [edge_attr]

        # ~~~~ Fill node mask:

          if return_mask: 
            
            perm_global = perms[0] 
            mask[perm_global] = 1
            for i in range(1,self.depth):
                perm_global = perm_global[perms[i]]
                mask[perm_global] = i+1
          data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))
          

        return data_list, batches , xs, positions, edge_attrs, edge_indices, edge_masks,perms
    def input_dict(self):
        a = { 'hidden_channels' : self.hidden_channels, 
              'in_channels_node' : self.in_channels_node, 
              'in_channels_edge' : self.in_channels_edge,
               'out_channels' : self.out_channels, 
                'n_mlp_mp' : self.n_mlp_mp, 
                'n_mlp_encode' : self.n_mlp_encode,
                'ae_dim':self.ae_dim,
                'n_mp_down' : self.n_mp_down, 
                'n_mp_up' : self.n_mp_up, 
                'pool_num' : self.pool_num, 
                
                'encoding_dim':self.encoding_dim,
                
                
                'depth' : self.depth, 
                'act' : self.act, 
                
                'name' : self.name }

        return a

    def reset_parameters(self):
        # Node encoding
        for module in self.node_encode:
            module.reset_parameters()

        # Edge encoding 
        for module in self.edge_encode:
            module.reset_parameters()

        

        # Pooling: 
        for module in self.pools:
            module.reset_parameters() 

        for modulelist_level in self.attn_score_down:
            for module in modulelist_level:
                module.reset_parameters()

    
        
        for modulelist_level in self.edge_down_mps:
            for modulelist_mp in modulelist_level:
                for module in modulelist_mp: 
                    module.reset_parameters()

        # Down Message passing, node update 
        for modulelist_level in self.node_down_mps:
            for modulelist_mp in modulelist_level:
                for module in modulelist_mp: 
                    module.reset_parameters()

                
        for modulelist_level in self.autoencode_down:
            for modulelist_mp in modulelist_level:
                for module in modulelist_mp:
                    module.reset_parameters() 




class GNN_Decode(torch.nn.Module):
    def __init__(self,
                 hidden_channels: int, 
                 in_channels_node : int, 
                 in_channels_edge : int, 
                 out_channels : int, 
                 encoding_dim:int,
                 n_mlp_mp : int, 
                 n_mlp_encode:int,
                 ae_dim:List[int],
                 n_mp_down : List[int], 
                 n_mp_up : List[int], 
                 pool_num:List[int],
                 act : Optional[Callable] = F.elu, 
                 name : Optional[str] = 'mmp_layer',
                ):
        super().__init__()
        self.edge_aggregator = EdgeAggregation()
        self.hidden_channels = hidden_channels
        self.in_channels_node = in_channels_node
        self.in_channels_edge = in_channels_edge
        self.out_channels = out_channels
        self.act = act
        self.n_mlp_encode = n_mlp_encode    # number of MLP layers in node/edge encoding stage 
        self.n_mlp_decode = n_mlp_encode 
        
        self.n_mlp_mp = n_mlp_mp # number of MLP layers in node/edge update functions used in message passing blocks
        self.ae_dim=ae_dim
        self.n_mp_down = n_mp_down # number of message passing blocks in downsampling path 
        self.n_mp_up = n_mp_up # number of message passing blocks in upsampling path  
        
        self.depth = len(self.n_mp_up)-1 # depth of u net 
        self.pool_num=pool_num #[1000,500,200] 
        self.encoding_dim=encoding_dim
        self.name = name


        self.edge_up_mps = torch.nn.ModuleList() 
        self.edge_up_norms = torch.nn.ModuleList()
        for m in range(len(n_mp_up)):
            n_mp=n_mp_up[m]
            edge_mp = torch.nn.ModuleList()
            edge_mp_norm = torch.nn.ModuleList()
            for i in range(n_mp):
                temp = torch.nn.ModuleList()
                for j in range(self.n_mlp_mp):
                    if j == 0:
                        input_features = hidden_channels*3
                        output_features = hidden_channels
                    else:
                        input_features = hidden_channels
                        output_features = hidden_channels
                    temp.append( nn.Linear(input_features, output_features) )
                edge_mp.append(temp)
                edge_mp_norm.append( nn.LayerNorm(output_features) )
            self.edge_up_mps.append(edge_mp)
            self.edge_up_norms.append(edge_mp_norm)


        # ~~~~ UPWARD Message Passing
        self.node_up_mps = torch.nn.ModuleList()
        self.node_up_norms = torch.nn.ModuleList()

        # Loop through levels: 
        for m in range(len(n_mp_up)):
            n_mp = n_mp_up[m]
            node_mp = torch.nn.ModuleList()
            node_mp_norm = torch.nn.ModuleList()

            # Loop through message passing steps per level 
            for i in range(n_mp):
                temp = torch.nn.ModuleList()

                # Loop through layers in MLP
                for j in range(self.n_mlp_mp):
                    if m ==(len(n_mp_up)-1):
                        if j==0:
                            input_features = hidden_channels*2
                            output_features = hidden_channels 
                        else:
                            input_features = hidden_channels
                            output_features = hidden_channels
                        temp.append( nn.Linear(input_features, output_features) )
                    else:
                        if j == 0:
                            input_features = hidden_channels*2+encoding_dim
                            output_features = hidden_channels 
                        else:
                            input_features = hidden_channels
                            output_features = hidden_channels
                        temp.append( nn.Linear(input_features, output_features) )
                node_mp.append(temp)
                node_mp_norm.append( nn.LayerNorm(output_features))

            self.node_up_mps.append(node_mp)
            self.node_up_norms.append(node_mp_norm)


        self.autoencode_up = torch.nn.ModuleList()
        self.attn_score_up = torch.nn.ModuleList()

        for m in range(len(n_mp_up)-1):
            n_ae=n_mp_up[m] 
            n_node_num=pool_num[self.depth-m-1]
            node_ae = torch.nn.ModuleList()
            n_attn = torch.nn.ModuleList()

            for i in range(n_ae):
                temp = torch.nn.ModuleList()
                for j in range(len(self.ae_dim)):
                    if j==0:
                        input_features = hidden_channels*n_node_num
                        output_features = self.ae_dim[j]
                    else:
                        input_features = self.ae_dim[j-1]
                        output_features = self.ae_dim[j]
                    temp.append( nn.Linear(input_features, output_features) )
                node_ae.append(temp)
                n_attn.append(Global_attention(hidden_channels, encoding_dim))
            self.autoencode_up.append(node_ae)
            self.attn_score_up.append(n_attn)


       
            


        # ~~~~ Node-wise decoder
        self.node_decode = torch.nn.ModuleList() 
        for i in range(self.n_mlp_decode): 
            if i == self.n_mlp_decode - 1:
                input_features = hidden_channels 
                output_features = out_channels
            else:
                input_features = hidden_channels 
                output_features = hidden_channels 
            self.node_decode.append( nn.Linear(input_features, output_features) )

        # ~~~~ Reset params upon initialization
        self.reset_parameters()

    def forward(self, data, batches , xs, positions, edge_attrs, edge_indices, edge_masks,perms, batch=None, return_mask=True):
        
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        m = 0
         
        for i in range(self.n_mp_up[m]):
              x_own = x[edge_index[0,:], :]
              x_nei = x[edge_index[1,:], :]
              edge_attr_t = torch.cat((x_own, x_nei, edge_attr), axis=1)
            
            # edge update mlp
              for g in range(self.n_mlp_mp):
                  
                  edge_attr_t = self.edge_up_mps[m][i][g](edge_attr_t) 
                  if g < self.n_mlp_mp - 1:
                      edge_attr_t = self.act(edge_attr_t)

              edge_attr = edge_attr + edge_attr_t
              edge_attr = self.edge_up_norms[m][i](edge_attr)

              # edge aggregation 
              edge_agg = self.edge_aggregator(x, edge_index, edge_attr)

              x_ae=mp_ae_dataprocess(x) 
              #print(x.shape,x_ae.shape)
              for g in range(len(self.ae_dim)):
                  x_ae = self.autoencode_up[m][i][g](x_ae) 
                  if g <len(self.ae_dim) - 1:
                      x_ae = self.act(x_ae)

              #print(x.shape,x_ae.shape)
              #不同节点对全局特征向量的重要有差异
              g=self.attn_score_up[m][i](x, x_ae)

              x_t = torch.cat((x, edge_agg,g), axis=1)

             

              # node update mlp
              for g in range(self.n_mlp_mp):
                  x_t = self.node_up_mps[m][i][g](x_t)
                  if g < self.n_mlp_mp - 1:
                      x_t = self.act(x_t) 
            
              x = x + x_t
              x = self.node_up_norms[m][i](x)

          
        # upward cycle
        for m in range(self.depth):
            # Get the fine level index
            fine = self.depth - 1 - m

            # Get the batch
            batch = batches[fine]

            # Get node features and edge features on fine level
            res = xs[fine]
            pos = positions[fine]
            res_edge = edge_attrs[fine]

            # Get edge index on fine level
            edge_index = edge_indices[fine]

            # Upsample edge features
            edge_mask = edge_masks[fine]
            up_edge = torch.zeros_like(res_edge)
            up_edge[edge_mask] = edge_attr
            edge_attr = up_edge

            # Upsample node features
            # get node assignments on fine level
            perm = perms[fine]
            up = torch.zeros_like(res)
            up[perm] = x
            x = up
            

            # Message passing on new upsampled graph
            for i in range(self.n_mp_up[m+1]):
                x_own = x[edge_index[0,:], :]
                x_nei = x[edge_index[1,:], :]
                edge_attr_t = torch.cat((x_own, x_nei, edge_attr), axis=1)
            
            # edge update mlp
                for g in range(self.n_mlp_mp):
                    edge_attr_t = self.edge_up_mps[m+1][i][g](edge_attr_t) 
                    if g < self.n_mlp_mp - 1:
                        edge_attr_t = self.act(edge_attr_t)

                edge_attr = edge_attr + edge_attr_t
                edge_attr = self.edge_up_norms[m+1][i](edge_attr)

              # edge aggregation 
                edge_agg = self.edge_aggregator(x, edge_index, edge_attr)

                if m< (self.depth-1) :
                    x_ae=mp_ae_dataprocess(x) 
                    #print(x.shape,x_ae.shape)
                    for g in range(len(self.ae_dim)):
                        x_ae = self.autoencode_up[m+1][i][g](x_ae) 
                        if g <len(self.ae_dim) - 1:
                            x_ae = self.act(x_ae)

              
                    #print(x.shape,x_ae.shape)
                    #不同节点对全局特征向量的重要有差异
                    g=self.attn_score_up[m+1][i](x, x_ae)

                    x_t = torch.cat((x, edge_agg,g), axis=1)
                else:
                    x_t = torch.cat((x, edge_agg), axis=1)


              # node update mlp
                for g in range(self.n_mlp_mp):
                    x_t = self.node_up_mps[m+1][i][g](x_t)
                    if g < self.n_mlp_mp - 1:
                        x_t = self.act(x_t) 
            
                x = x + x_t
                x = self.node_up_norms[m+1][i](x)
            # ~~~~ Node decoder
          
        for i in range(self.n_mlp_decode):
            x = self.node_decode[i](x) 
            if i < self.n_mlp_decode - 1:
                x = self.act(x)
            else:
                x = x
         

        return x

    def input_dict(self):
        a = { 'hidden_channels' : self.hidden_channels, 
              'in_channels_node' : self.in_channels_node, 
              'in_channels_edge' : self.in_channels_edge,
               'out_channels' : self.out_channels, 
                'n_mlp_mp' : self.n_mlp_mp, 
                'n_mlp_encode' : self.n_mlp_encode,
                'ae_dim':self.ae_dim,
                'n_mp_down' : self.n_mp_down, 
                'n_mp_up' : self.n_mp_up, 
                'pool_num' : self.pool_num, 
                
                'encoding_dim':self.encoding_dim,
                
                
                'depth' : self.depth, 
                'act' : self.act, 
                
                'name' : self.name}

        return a

    def reset_parameters(self):


        # Node decoder: 
        for module in self.node_decode:
            module.reset_parameters()


        for modulelist_level in self.attn_score_up:
            for module in modulelist_level:
                module.reset_parameters()

        


        # Up message passing, edge update 
        for modulelist_level in self.edge_up_mps:
            for modulelist_mp in modulelist_level:
                for module in modulelist_mp: 
                    module.reset_parameters()

        # Up message passing, node update 
        for modulelist_level in self.node_up_mps:
            for modulelist_mp in modulelist_level:
                for module in modulelist_mp: 
                    module.reset_parameters() 

        for modulelist_level in self.autoencode_up:
            for modulelist_mp in modulelist_level:
                for module in modulelist_mp: 
                    
                    module.reset_parameters()
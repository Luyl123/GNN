import torch
import torch.nn as nn
from torch.nn import Linear, Sequential, LayerNorm, ReLU


import torch.optim as optim
from torch_geometric.data import DataLoader
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Data


from tqdm import trange
import os
import copy
import matplotlib.pyplot as plt
import pandas as pd


class self_transformer(torch.nn.Module):
    def __init__(self, args,n_edge_num, emb=False):
        super(self_transformer, self).__init__()
       
        self.d_k=args.d_k
        self.d_v=args.d_v
        self.input_shape=args.input_shape 
        self.batchsize=args.batchsize
        self.seq_len=args.seq_len
        self.n_heads=args.n_heads
        self.n_layers=args.n_layers
        self.ff_dim=args.ff_dim
        self.n_graph_encode=args.n_graph_encode
        self.pool_num=args.pool_num
        self.hidden_dim=args.hidden_dim
        
        self.n_edge_num=n_edge_num

        
     
        self.graph_encode = torch.nn.ModuleList()
        n_node_num=self.pool_num[-1]
        
        for j in range(len(self.n_graph_encode)):
                if j==0:
                    input_features = self.hidden_dim*n_node_num+self.n_edge_num*self.hidden_dim
                    output_features = self.n_graph_encode[j]
                else:
                    input_features = self.n_graph_encode[j-1]
                    output_features = self.n_graph_encode[j]
                self.graph_encode.append( nn.Linear(input_features, output_features) )
        
        self.transformer=Transformer(self.d_k, self.d_v,self.input_shape ,self.n_heads,self.n_layers,self.ff_dim,self.seq_len, self.n_graph_encode[-1])
        
        self.graph_decode = torch.nn.ModuleList()
        for i in range(len(self.n_graph_encode)):
            if i<(len(self.n_graph_encode)-1):
                input_features = self.n_graph_encode[(len(self.n_graph_encode)-1-i)]
                output_features = self.n_graph_encode[(len(self.n_graph_encode)-1-(i+1))]
            else:
                input_features = self.n_graph_encode[(len(self.n_graph_encode)-1-i)]
                output_features=self.hidden_dim*n_node_num+self.n_edge_num*self.hidden_dim
            self.graph_decode.append( nn.Linear(input_features, output_features) )
        

    def forward(self,data,mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge):
        """
        Encoder encodes graph (node/edge features) into latent vectors (node/edge embeddings)
        The return of processor is fed into the processor for generating new feature vectors
        """

        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr   #batch_size*per_t个图拼出来的
        
        

        x = normalize(x,mean_vec_x,std_vec_x)
        edge_attr=normalize(edge_attr,mean_vec_edge,std_vec_edge)

        x=x.reshape((self.batchsize*self.seq_len),-1)  
        aa=x.shape[1]
        #print(x.shape,aa) #点数乘维数
        edge_attr=edge_attr.reshape((self.batchsize*self.seq_len),-1) #边数乘维数
        
        #print(edge_attr.shape)

        z=torch.cat((x,edge_attr),dim=1)

        #print(z.shape)
        for i in range(len(self.n_graph_encode)):
            z = self.graph_encode[i](z) 
            if i <len(self.n_graph_encode) - 1:
                 z = F.elu(z)
        #print(x.shape)     #(batch_size,self.n_graph_encode[-1])
        z=z.reshape(-1,self.seq_len,z.shape[1])        #预测的btach_size比信息传递的batch_size小self.seq_len倍
        #x=x.reshape(seq_len,int(x.shape[0]/seq_len),-1)
        #x=torch.cat((x[0],x[1]),dim=-1)
        #x=x.reshape(int(x.shape[0]),int(seq_len),-1)
        # step 2:transformer
        
        z=self.transformer(z)     #(batch_size/self.seq_len,1,self.n_graph_encode[-1])
        #print(x.shape)
        #x=torch.squeeze(x)    
        #print(z.shape)

        for i in range(len(self.n_graph_encode)):
            z = self.graph_decode[i](z) 
            if i <len(self.n_graph_encode) - 1:
                 z = F.elu(z)
        #print(z.shape)          #(batch_size,点数*hidden_dim)
        x=z[:,:aa]
        #print(x.shape)
        edge_attr=z[:,aa:]
        #print(edge_attr.shape)
        x=x.reshape(-1,self.hidden_dim)

        edge_attr=edge_attr.reshape(-1,self.hidden_dim)
        #print(x.shape)

        


        return x,edge_attr



def normalize(to_normalize,mean_vec,std_vec):
    return (to_normalize-mean_vec)/std_vec

def unnormalize(to_unnormalize,mean_vec,std_vec):
    return to_unnormalize*std_vec+mean_vec

class Time2Vector(torch.nn.Module):
    def __init__(self, seq_len, **kwargs):
      super(Time2Vector, self).__init__()
      self.seq_len = seq_len            #所需要的时间4推1
      '''Initialize weights and biases with shape (batch, seq_len)'''
      self.weights_linear=nn.Parameter(torch.zeros(int(self.seq_len),),requires_grad=True)
      nn.init.uniform_(self.weights_linear)
      
      self.bias_linear=nn.Parameter(torch.zeros(int(self.seq_len),),requires_grad=True)
      nn.init.uniform_(self.bias_linear)
      
      self.weights_periodic=nn.Parameter(torch.zeros(int(self.seq_len),),requires_grad=True)
      nn.init.uniform_(self.weights_periodic)

      self.bias_periodic=nn.Parameter(torch.zeros(int(self.seq_len),),requires_grad=True)
      nn.init.uniform_(self.bias_periodic)

    def forward(self, x):
      '''Calculate linear and periodic time features'''
      
      x=torch.mean(x,dim=-1)
      
      time_linear = self.weights_linear * x + self.bias_linear # Linear time feature
      
      time_linear=torch.unsqueeze(time_linear,dim=-1)
      
      time_periodic = torch.sin(torch.mul(x, self.weights_periodic) + self.bias_periodic) #对应位置相乘
      
      time_periodic=torch.unsqueeze(time_periodic,dim=-1)
      
      return torch.cat([time_linear, time_periodic], dim=-1) # shape = (batch, seq_len, 2)
    

class SingleAttention(torch.nn.Module):
    def __init__(self, d_k, d_v,input_shape):
      super(SingleAttention, self).__init__()
      self.d_k = d_k
      self.d_v = d_v
      self.input_shape=input_shape

      self.query = nn.Linear(self.input_shape, self.d_k,
                               bias=False)
      
      torch.nn.init.xavier_uniform_(self.query.weight)
      
      
      
      
      self.key=nn.Linear(self.input_shape,self.d_k,bias=False)
      torch.nn.init.xavier_uniform_(self.key.weight)
      
      self.value=nn.Linear(self.input_shape,self.d_v,bias=False)
      torch.nn.init.xavier_uniform_(self.value.weight)
      
    def forward(self, inputs): # inputs = (in_seq, in_seq, in_seq)
      q = self.query(inputs[0])
      
      k = self.key(inputs[1])
      
      attn_weights = torch.matmul(q, torch.transpose(k,1,2))
      
      attn_weights=attn_weights/np.sqrt(self.d_k)
      attn_weights = F.softmax(attn_weights, dim=-1)
      
      v = self.value(inputs[2])
      attn_out = torch.matmul(attn_weights, v)
      
      return attn_out    

class MultiAttention(torch.nn.Module):
    def __init__(self, d_k, d_v, input_shape,n_heads):
      super(MultiAttention, self).__init__()
      self.d_k = d_k
      self.d_v = d_v
      self.input_shape=input_shape
      self.n_heads = n_heads
      self.attn_heads = nn.ModuleList()

      
      for n in range(self.n_heads):
        
        self.attn_heads.append(SingleAttention(self.d_k, self.d_v,self.input_shape))  
      
      # input_shape[0]=(batch, seq_len, 7), input_shape[0][-1]=7 
      self.linear =nn.Linear(int(self.n_heads*self.d_v), 
                          self.input_shape, 
                          bias=True)
      
      torch.nn.init.xavier_uniform_(self.linear.weight)
      self.linear.bias.data.fill_(0)
      
    def forward(self, inputs):
      attn = [self.attn_heads[i](inputs) for i in range(self.n_heads)]
      
      concat_attn = torch.cat(attn, dim=-1)
      
      multi_linear = self.linear(concat_attn)
      return multi_linear   

class TransformerEncoder(torch.nn.Module):
    def __init__(self, d_k, d_v, input_shape,n_heads, ff_dim, dropout=0.1, **kwargs):
      super(TransformerEncoder, self).__init__()
      self.d_k = d_k
      self.d_v = d_v
      self.input_shape=input_shape
      self.n_heads = n_heads
      self.ff_dim = ff_dim
      self.attn_heads = list()
      self.dropout_rate = dropout

      self.attn_multi = MultiAttention(self.d_k, self.d_v, self.input_shape,self.n_heads)
      self.attn_dropout = torch.nn.Dropout(self.dropout_rate)
      self.attn_normalize = torch.nn.LayerNorm(self.input_shape, eps=1e-6) #按照最后的维度做标准化

      
      self.ff_conv1D_1=torch.nn.Conv1d(self.input_shape, self.ff_dim, kernel_size=1)
      
      # input_shape[0]=(batch, seq_len, 7), input_shape[0][-1] = 7 
      self.ff_conv1D_2 = torch.nn.Conv1d(self.ff_dim,self.input_shape, kernel_size=1) 

      self.ff_dropout = torch.nn.Dropout(self.dropout_rate)
      self.ff_normalize = torch.nn.LayerNorm(self.input_shape, eps=1e-6)   
      
    def forward(self, inputs): # inputs = (in_seq, in_seq, in_seq)
      attn_layer = self.attn_multi(inputs)
      
      attn_layer = self.attn_dropout(attn_layer)
      
      attn_layer = self.attn_normalize(inputs[0] + attn_layer)
      attn_layer=attn_layer.permute(0,2,1)
      ff_layer = self.ff_conv1D_1(attn_layer)
      
      ff_layer=F.relu(ff_layer)
      ff_layer = self.ff_conv1D_2(ff_layer)
      ff_layer=F.relu(ff_layer)
      ff_layer = self.ff_dropout(ff_layer)
      attn_layer=attn_layer.permute(0,2,1)
      ff_layer=ff_layer.permute(0,2,1) 
      ff_layer = self.ff_normalize(attn_layer + ff_layer)
      return ff_layer 
 

class Transformer(torch.nn.Module):
    """docstring for Transformer"""
    def __init__(self,d_k, d_v,input_shape ,n_heads,n_layers,ff_dim,seq_len, encoding_dim):
      super(Transformer, self).__init__()
      self.d_k = d_k
      self.d_v = d_v
      self.input_shape=input_shape
      self.n_heads = n_heads
      self.n_layers=n_layers
      self.ff_dim = ff_dim
      self.seq_len = seq_len    #几个时间段推一个时间段
      self.ae_encoding_dim=encoding_dim

      self.time_embedding = Time2Vector(self.seq_len)
      self.attn_layers = nn.ModuleList()
      for n in range(self.n_layers):
          self.attn_layers.append(TransformerEncoder(self.d_k, self.d_v, self.input_shape ,self.n_heads, self.ff_dim))
      self.flatten=torch.nn.Flatten()
      self.dropout_1=torch.nn.Dropout(0.2)
      self.dense_1=torch.nn.Linear(self.seq_len*self.input_shape,64,bias=True)
      self.relu_1=torch.nn.ReLU()
      self.dropout_2=torch.nn.Dropout(0.2)
      self.dense_2=torch.nn.Linear(64,encoding_dim,bias=True)
    def forward(self,in_seq):
        '''Construct model'''
        x = self.time_embedding(in_seq)
        
        x = torch.cat([in_seq, x],dim=-1) #(batch,4,22+2)
        for i in range(self.n_layers):
            
            x = self.attn_layers[i]((x,x,x))
        
        x = self.flatten(x)
        x=self.dropout_1(x)
        x=self.dense_1(x)
        x=self.relu_1(x)
        x=self.dropout_2(x)
        out=self.dense_2(x)
        
        return out 


def Mulit_attention_train(data_,data_label, device,data_stats_list, args,edge_num):
    '''
    Performs a training loop on the dataset for MeshGraphNets. Also calls
    test and validation functions.
    '''

    df = pd.DataFrame(columns=['epoch','train_loss','test_loss'])

    #Define the model name for saving
    model_name='Multi-attention_model_nl'+'_bs'+str(args.batchsize) + \
               '_hd'+str(args.hidden_dim)+'_ep'+str(args.epochs)+'_wd' + \
               '_shuff_'+str(args.shuffle)+'_dk'+str(args.d_k)+'_heads'+str(args.n_heads)
    
    
    data_transform_loader_train=DataLoader(data_[:int(args.train_size*args.seq_len)], batch_size=int(args.seq_len*args.batchsize), shuffle=False) 
    data_transform_loader_test=DataLoader(data_[int(args.train_size*args.seq_len):], batch_size=int(args.seq_len*args.batchsize), shuffle=False)

    label_transform_loader_train=DataLoader(data_label[:args.train_size], batch_size=args.batchsize, shuffle=False)
    label_transform_loader_test=DataLoader(data_label[args.train_size:], batch_size=args.batchsize, shuffle=False)

    
    #torch_geometric DataLoaders are used for handling the data of lists of graphs
    
    #The statistics of the data are decomposed
    [mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge] = data_stats_list
    (mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge)=(mean_vec_x.to(device),
        std_vec_x.to(device),mean_vec_edge.to(device),std_vec_edge.to(device))
    
    model = self_transformer(args,edge_num).to(device)
    #model.load_state_dict(torch.load('/root/data1/d_EMD/mgnn_transformerbest_models/g_trans_model_nl2_bs10_hd25_ep4000_wd_shuff_False_dk28_heads6.pt'))
    
    '''optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #weight_decay=1e-4
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                patience=1000, threshold=0.0000001, threshold_mode='rel',
                                cooldown=50, min_lr=0.000001, eps=1e-08, verbose=True) '''
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.01) #weight_decay=1e-4
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                patience=100, threshold=0.000000001, threshold_mode='rel',
                                cooldown=0, min_lr=0.000001, eps=1e-08, verbose=True)

    # train
    losses = []
    test_losses = []
    criterion = torch.nn.MSELoss()
    best_test_loss = np.inf  #初始化为正无穷
    best_model = None
    for epoch in trange(args.epochs, desc="Training", unit="Epochs"):
        total_loss = 0
        
        model.train()
        num_loops=0
        for i,batch in enumerate(zip(data_transform_loader_train,label_transform_loader_train)):
            #Note that normalization must be done before it's called. The unnormalized
            #data needs to be preserved in order to correctly calculate the loss

            data,lable=batch
            data=data.to(device)
            lable=lable.to(device)
            
            optimizer.zero_grad()         #zero gradients each time
            
            pred_x ,pred_edge_attr= model(data,mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge)
            labels_x = normalize(lable.x,mean_vec_x,std_vec_x)
            labels_edge = normalize(lable.edge_attr,mean_vec_edge,std_vec_edge)
            loss_1 = criterion(pred_x, labels_x)
            loss_2 = criterion(pred_edge_attr, labels_edge)
            loss=(0.7)*loss_1+(0.3)*loss_2

            
            
            #loss = model.loss(pred,lable,mean_vec_y,std_vec_y)
            loss.backward()         #backpropagate loss
            optimizer.step()
            total_loss += loss.item()
           
            num_loops+=1
            
            
        total_loss /= num_loops
        

        losses.append(total_loss)#记录每个epoch的loss平均值

        #Every tenth epoch, calculate acceleration test loss and velocity validation loss
        test_loss= Mulit_attention_test(model,data_transform_loader_test,label_transform_loader_test,device, criterion,mean_vec_x,
                                              std_vec_x,mean_vec_edge,std_vec_edge)
            
        test_losses.append(test_loss.item())
        scheduler.step(test_loss)

            # saving model
        if not os.path.isdir( args.checkpoint_dir ):#如果不是目录文件，将执行下面的代码，新建一个目录文件
                os.mkdir(args.checkpoint_dir)

        PATH = os.path.join(args.checkpoint_dir, model_name+'.csv')
        df.to_csv(PATH,index=False)

            #save the model if the current one is better than the previous best
        if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_model = copy.deepcopy(model)

        
        #df = df.append({'epoch': epoch, 'train_loss': losses[-1], 'test_loss': test_losses[-1]}, ignore_index=True)
        df = pd.concat([df, pd.DataFrame({'epoch': epoch, 'train_loss': losses[-1], 'test_loss': test_losses[-1]}, index=[0])])
        if(epoch%100==0):
           
            if(args.save_best_model):

                PATH = os.path.join(args.checkpoint_dir, model_name+'.pt')
                torch.save(best_model.state_dict(), PATH )
        if(epoch%100==0):
            print("train loss", str(round(total_loss,6)), "test loss", str(round(test_loss.item(),6)))
        

    return test_losses, losses, best_model, best_test_loss


def Mulit_attention_test(model,data_transform_loader_test, label_transform_loader_test,device,
                            criterion,mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge,
                           save_model_preds=False, model_type=None):
   

    '''
    Calculates test set losses and validation set errors.
    '''
    model.eval()

    loss_total = 0.0
    
    num_loops = 0
    with torch.no_grad():
        for i,batch in enumerate(zip(data_transform_loader_test, label_transform_loader_test)):
            data,lable=batch
            data=data.to(device)
            lable=lable.to(device)
            
        

            #calculate the loss for the model given the test set
            pred_x,pred_edge_attr = model(data,mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge)
            
            labels_x = normalize(lable.x,mean_vec_x,std_vec_x)
            labels_edge = normalize(lable.edge_attr,mean_vec_edge,std_vec_edge)
            loss_1 = criterion(pred_x, labels_x)
            loss_2 = criterion(pred_edge_attr, labels_edge)
            loss=loss_1+(0.3)*loss_2
            
            num_loops=num_loops+1


            
            
            loss_total += loss
            

            #loss += test_model.loss(pred, lable,mean_vec_y,std_vec_y)
            
        # if velocity is evaluated, return velo_rmse as 0
        loss_total /= num_loops
    return loss_total


    

def Mulit_attention_pred(data,model,mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge):
    model.eval()
    
    
    with torch.no_grad():
        
       
        x,edge_attr= model(data,mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge)
        x = unnormalize(x,mean_vec_x,std_vec_x)
        edge_attr = unnormalize(edge_attr,mean_vec_edge,std_vec_edge)
            
            
    return x,edge_attr




def save_plots(args, losses, test_losses):
    
    model_name='Multi-attention_model_nl'+'_bs'+str(args.batchsize) + \
               '_hd'+str(args.hidden_dim)+'_ep'+str(args.epochs)+'_wd' + \
               '_shuff_'+str(args.shuffle)+'_dk'+str(args.d_k)+'_heads'+str(args.n_heads)

    if not os.path.isdir(args.postprocess_dir):
        os.mkdir(args.postprocess_dir)

    PATH = os.path.join(args.postprocess_dir, model_name + '.pdf')

    f = plt.figure()
    plt.title('Losses Plot')
    plt.plot(losses, label="training loss" + " - " + args.model_type)
    plt.plot(test_losses, label="test loss" + " - " + args.model_type)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend()
    plt.show()
    f.savefig(PATH, bbox_inches='tight')  






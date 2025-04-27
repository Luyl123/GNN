import os
import tensorflow.compat.v1 as tf
import pandas as pd
import meshio
import torch
from matplotlib import tri as mtri
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch_geometric.data import Data
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import trange
from sklearn.metrics import mean_squared_error
import math
import vtk

def read_vtk_file(filename):
    # Read the VTK file
    reader=vtk.vtkPolyDataReader()
    reader.SetFileName(filename)  # SetFileName设置要读取的vtk文件
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.ReadAllTensorsOn()
    reader.Update()

    vtkdata=reader.GetOutput()  # GetOutput获取文件的数据
    #print(vtkdata)
    num_points=vtkdata.GetNumberOfPoints()  # GetNumberOfPoint获取点的个数
    #print(num_points)
    num_cells=vtkdata.GetNumberOfCells()
    #print(num_cells)
    
    for i in range(num_points):
     if i==0:
        point_coords=torch.tensor(vtkdata.GetPoint(i)[0:2]).type(torch.float)
     else:
        b=torch.tensor(vtkdata.GetPoint(i)[0:2]).type(torch.float)
        point_coords=torch.vstack((point_coords,b))  
    #print(len(point_coords))
    #print(point_coords[0]) 
    for i in range(num_cells):
        cell = vtkdata.GetCell(i)
        
        cell_points = [cell.GetPointId(j) for j in range(cell.GetNumberOfPoints())]
        cell_points=np.array(cell_points).reshape(-1,3)
        if i ==0:
            cells=cell_points
        else:
            c=cell_points
            cells=np.vstack((cells,c))     
    #print(len(cells))
    #print(cells[0])

    point_data =vtkdata.GetPointData()
    
    velocity_array = point_data.GetArray('U')  # Replace 'velocity' with the actual array name
    for i in range(num_points):
       if i==0:
          velocities=torch.tensor(velocity_array.GetTuple(i)[0:2]).type(torch.float)
       else:
          a=torch.tensor(velocity_array.GetTuple(i)[0:2]).type(torch.float)
          velocities=torch.vstack((velocities,a))
    #print(len(velocities))
    #print(velocities[1])
    
    return point_coords, cells, velocities


def get_vtk_num(path):
	# count the number of vtu files
		f_list = os.listdir(path)
		vtk_num = 0
		for i in f_list:
			if os.path.splitext(i)[1] == '.vtk':
				vtk_num = vtk_num+1
		return vtk_num

def quad_to_edges(faces):
  
  # collect edges from triangles
  edges = tf.concat([faces[:, 0:2],
                     faces[:, 1:3],faces[:, 2:4],
                     tf.stack([faces[:, 3], faces[:, 0]], axis=1)], axis=0)
  # those edges are sometimes duplicated (within the mesh) and sometimes
  # single (at the mesh boundary).
  # sort & pack edges as single tf.int64
  #print(edges.shape)
  receivers = tf.reduce_min(edges, axis=1)
  senders = tf.reduce_max(edges, axis=1)
  packed_edges = tf.bitcast(tf.stack([senders, receivers], axis=1), tf.int64)
  #print(packed_edges.shape)
  # remove duplicates and unpack
  df = pd.DataFrame(packed_edges)
  unique_edges = df.drop_duplicates()
  unique_edges = tf.convert_to_tensor(unique_edges)
  senders, receivers = tf.unstack(unique_edges, axis=1)
  # create two-way connectivity
  return (tf.concat([senders, receivers], axis=0),
          tf.concat([receivers, senders], axis=0))

def triangles_to_edges(faces):
  """Computes mesh edges from triangles.
     Note that this triangles_to_edges method was provided as part of the
     code release for the MeshGraphNets paper by DeepMind, available here:
     https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets
  """

  # collect edges from triangles
  edges = tf.concat([faces[:, 0:2],
                     faces[:, 1:3],
                     tf.stack([faces[:, 2], faces[:, 0]], axis=1)], axis=0)
  # those edges are sometimes duplicated (within the mesh) and sometimes
  # single (at the mesh boundary).
  # sort & pack edges as single tf.int64
  #print(edges.shape)
  receivers = tf.reduce_min(edges, axis=1)
  senders = tf.reduce_max(edges, axis=1)
  packed_edges = tf.bitcast(tf.stack([senders, receivers], axis=1), tf.int64)
  #print(packed_edges.shape)
  # remove duplicates and unpack
  df = pd.DataFrame(packed_edges)
  unique_edges = df.drop_duplicates()
  unique_edges = tf.convert_to_tensor(unique_edges)
  senders, receivers = tf.unstack(unique_edges, axis=1)
  # create two-way connectivity
  return (tf.concat([senders, receivers], axis=0),
          tf.concat([receivers, senders], axis=0))

def get_dataset(number_trajectories,satart_number_ts,number_ts,per_t,batch_size,dataset_dir,path):
  data_list = []

  for i in range(number_trajectories):

    if os.path.isdir(path+str(i)):
      path_k=path+str(i)
      print(path_k)
      vtk_num=get_vtk_num(path_k)
      print(vtk_num)
      for ts in range(satart_number_ts,vtk_num):
        if(ts==number_ts+satart_number_ts):
          break

        if((ts-satart_number_ts)%(batch_size*per_t)==0):

          for i in range(batch_size*per_t):
            if i==0:
              point, cells, velocity=read_vtk_file(path_k+"/slice_"+str(ts+i)+".vtk")
              
              
              edges = triangles_to_edges(tf.convert_to_tensor(cells))
              edge_index = torch.cat( (torch.tensor(edges[0].numpy()).unsqueeze(0) ,
                         torch.tensor(edges[1].numpy()).unsqueeze(0)), dim=0).type(torch.long)
              
              u_i=point[edge_index[0]]
              u_j=point[edge_index[1]]
              u_ij=u_i-u_j
              u_ij_norm = torch.norm(u_ij,p=2,dim=1,keepdim=True)
              edge_attr = torch.cat((u_ij,u_ij_norm),dim=-1).type(torch.float)
              
              

            if i>=1:
              point1, cells1, velocity1=read_vtk_file(path_k+"/slice_"+str(ts+i)+".vtk")
              
              velocity=torch.vstack((velocity, velocity1))
              
          data_list.append(Data(x=velocity, edge_index=edge_index, edge_attr=edge_attr,
                                  cells=cells,mesh_pos=point))

  print("Done collecting data!")
  torch.save(data_list,dataset_dir+'/meshgraphnets_miniset'+str(per_t)+str(batch_size)+str(number_trajectories)+'traj'+str(satart_number_ts)+str(number_ts)+'ts_vis.pt')
  print("Done saving data!")
  print("Output Location: ", dataset_dir+'/meshgraphnets_miniset'+str(per_t)+str(batch_size)+str(number_trajectories)+'traj'+str(satart_number_ts)+str(number_ts)+'ts_vis.pt')

def get_EDstats(data_list):
    '''
    Method for normalizing processed datasets. Given  the processed data_list,
    calculates the mean and standard deviation for the node features, edge features,
    and node outputs, and normalizes these using the calculated statistics.
    '''

    #mean and std of the node features are calculated
    mean_vec_x=torch.zeros(data_list[0].x.shape[1:])#2
    std_vec_x=torch.zeros(data_list[0].x.shape[1:])#2

    #mean and std of the edge features are calculated
    mean_vec_edge=torch.zeros(data_list[0].edge_attr.shape[1:])#3
    std_vec_edge=torch.zeros(data_list[0].edge_attr.shape[1:])#3

    #Define the maximum number of accumulations to perform such that we do
    #not encounter memory issues
    max_accumulations = 10**6

    #Define a very small value for normalizing to
    eps=torch.tensor(1e-8)

    #Define counters used in normalization
    num_accs_x = 0
    num_accs_edge=0


    #Iterate through the data in the list to accumulate statistics
    for dp in data_list:

        #Add to the
        
        mean_vec_x+=torch.sum(dp.x,dim=0)
        std_vec_x+=torch.sum(dp.x**2,dim=0)
        num_accs_x+=dp.x.shape[0]

        mean_vec_edge+=torch.sum(dp.edge_attr,dim=0)
        std_vec_edge+=torch.sum(dp.edge_attr**2,dim=0)
        num_accs_edge+=dp.edge_attr.shape[0]

        if(num_accs_x>max_accumulations or num_accs_edge>max_accumulations ):
            break

    mean_vec_x = mean_vec_x/num_accs_x
    std_vec_x = torch.maximum(torch.sqrt(std_vec_x/num_accs_x - mean_vec_x**2),eps)

    mean_vec_edge = mean_vec_edge/num_accs_edge
    std_vec_edge = torch.maximum(torch.sqrt(std_vec_edge/num_accs_edge - mean_vec_edge**2),eps)

    mean_std_list=[mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge]

    return mean_std_list

def test(model, loader, perms,criterion,per_t,batch_size,EDmean_vec_x,EDstd_vec_x,EDmean_vec_edge,EDstd_vec_edge):
    model.eval()
    mse_total = 0.0
    
    count = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    
    with torch.no_grad():
        for data in loader:
            data=data.to(device)
            
            out,  _ = model(data, per_t , perms,batch_size,EDmean_vec_x,EDstd_vec_x,EDmean_vec_edge,EDstd_vec_edge,batch=None, return_mask=False)
            data.x=(data.x-EDmean_vec_x)/EDstd_vec_x
            mse = criterion(out, data.x)
            
            mse_total += mse.detach().item()
            
            
            #print(torch.sum(data.x))
            count += 1
        mse_total = mse_total / count
        
    return mse_total

def train(N_epochs, model,perm_list,train_loader, test_loader,per_t,batch_size,EDmean_vec_x,EDstd_vec_x,EDmean_vec_edge,EDstd_vec_edge,pathED,pathD,pathE):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00025) #weight_decay=0.001
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                patience=5, threshold=0.0000001, threshold_mode='rel',
                                cooldown=0, min_lr=0.000001, eps=1e-08, verbose=True)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[250,450,650,850,950], gamma=0.5)     #500

    
    criterion = torch.nn.MSELoss()
    acc=torch.nn.MSELoss(reduction='sum')

    train_hist = np.zeros(N_epochs)
    test_hist = np.zeros(N_epochs)
    update_iter = 0
    #per_t=4   #改
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    
    for epoch in trange(N_epochs, desc="Training", unit="Epochs"):
        '''if epoch%100==0 and epoch>=300:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) #weight_decay=1e-4
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                patience=5, threshold=0.0000001, threshold_mode='rel',
                                cooldown=0, min_lr=0.000001, eps=1e-08, verbose=True)'''

       
        
    
        time_epoch = time.time()
        model.train()
        batch_count = 0
        train_mse = 0
        train_RMSE=0
        train_acc=0


        for step, data in enumerate(train_loader):
            
            data=data.to(device)
            perms=perm_list[0]
            out, _ = model(data, per_t , perms,batch_size,EDmean_vec_x,EDstd_vec_x,
                               EDmean_vec_edge,EDstd_vec_edge,batch=None, return_mask=False)
        
            data.x=(data.x-EDmean_vec_x)/EDstd_vec_x  #并不影响train_loader里面的数据，只是取出来的data数据改变了
            #lable=(data.x-EDmean_vec_x)/EDstd_vec_x
            #loss = criterion(out, lable)
            loss = criterion(out, data.x)
            train_acc_=acc(out, data.x)
            loss.backward()
            optimizer.step()     #一个batch一更新
            optimizer.zero_grad()
            train_mse += loss.item()
            #train_RMSE+=math.sqrt(loss.item())
            #print(torch.sum(data.x))
            train_acc+=(math.sqrt(train_acc_.item())/torch.norm(data.x, p=2))
            batch_count += 1
            update_iter += 1
            


        train_mse = train_mse/batch_count   #一个epoch的平均mse
        train_RMSE=math.sqrt(train_mse)
        train_acc=train_acc/batch_count

        test_mse = test(model, test_loader,perm_list[0] ,criterion,per_t,batch_size,EDmean_vec_x,EDstd_vec_x,EDmean_vec_edge,EDstd_vec_edge)

        time_epoch = time.time() - time_epoch
        #print(f'Epoch: {epoch:04d},\tTrain mse: {train_mse:.8f},\tTrain_RMSE: {train_RMSE:.8f},\tTrain_acc: {train_acc:.8f},\tTest mse: {test_mse:.8f},\tTime: {time_epoch:.8f}s')
        if(epoch%2==0):
            print(f'Epoch: {epoch:04d},\tTrain mse: {train_mse:.8f},\tTrain_RMSE: {train_RMSE:.8f},\tTrain_acc: {train_acc:.8f},\tTest mse: {test_mse:.8f},\tTime: {time_epoch:.8f}s')

        # Accumulate accuracies
        train_hist[epoch] = train_mse
        test_hist[epoch] = test_mse

        # Step scheduler
        scheduler.step(test_mse)    #检测test的loss的下降

    # Save model:
    model.to('cpu')
    #model_savepath = 'model.tar'
    '''save_dict = {   'state_dict' : model.state_dict(),
                    'input_dict' : model.input_dict(),
                    'train_loss' : train_hist,
                    'test_loss': test_hist
                }'''
    #torch.save(save_dict, model_savepath)
    print('Model.state_dict:')
    for param_tensor in model.state_dict():
        #打印 key value字典
        print(param_tensor,'\t',model.state_dict()[param_tensor].size())
    #pathED = '/root/data1/EMD/ED_state_dict.pt'    
    #pathED = '/root/data1/EMD/MP_ED_state_dict.pt'   #没加入全局信息
    torch.save(model.state_dict(), pathED)

    save_state={}
    for param_tensor in model.state_dict():
      if 'node_encode' in param_tensor:
        save_state.update({param_tensor:model.state_dict()[param_tensor]})
      
      if 'edge_encode' in param_tensor:
        save_state.update({param_tensor:model.state_dict()[param_tensor]})
      if 'down_mps' in param_tensor:
        save_state.update({param_tensor:model.state_dict()[param_tensor]})
      if 'down_norms' in param_tensor:
        save_state.update({param_tensor:model.state_dict()[param_tensor]})
      if 'autoencode_down' in param_tensor:
        save_state.update({param_tensor:model.state_dict()[param_tensor]})
      if 'attn_score_down' in param_tensor:
        save_state.update({param_tensor:model.state_dict()[param_tensor]})
      if 'pools' in param_tensor:
        save_state.update({param_tensor:model.state_dict()[param_tensor]})
    #pathE='/root/data1/EMD/Encode_MP_global_state_dict.pt'
    torch.save(save_state, pathE)

    save_state={}
    for param_tensor in model.state_dict():
      if 'up_mps' in param_tensor:
        save_state.update({param_tensor:model.state_dict()[param_tensor]})
      if 'up_norms' in param_tensor:
        save_state.update({param_tensor:model.state_dict()[param_tensor]})
      if 'autoencode_up' in param_tensor:
        save_state.update({param_tensor:model.state_dict()[param_tensor]})
      if 'attn_score_up' in param_tensor:
        save_state.update({param_tensor:model.state_dict()[param_tensor]})
      if 'node_decode' in param_tensor:
        save_state.update({param_tensor:model.state_dict()[param_tensor]})
    #pathD='/root/data1/EMD/Decode_MP_global_state_dict.pt'
    torch.save(save_state, pathD)
    return train_hist, test_hist

def save_DEplots(losses, test_losses,plot_dir):

    #plot_dir='/root/data1/EMD/loss/'
    
    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir)

    PATH = os.path.join(plot_dir,'ED.pdf')

    f = plt.figure()
    plt.title('Losses Plot')
    plt.plot(losses, label="training loss-Encode_decode"  )
    plt.plot(test_losses, label="test loss-Encode_decode" )
    #if (args.save_velo_val):
    #    plt.plot(velo_val_losses, label="velocity loss" + " - " + args.model_type)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend()
    plt.show()
    f.savefig(PATH, bbox_inches='tight')

def make_animation(gs, pred, evl, path, name , skip = 2, save_anim = True, plot_variables = False):
    '''
    input gs is a dataloader and each entry contains attributes of many timesteps.

    '''
    print('Generating velocity fields error...')
    fig, axes = plt.subplots(3, 1, figsize=(20, 16))
    num_steps = len(gs) # for a single trajectory
    #print(num_steps)
    num_frames = num_steps // skip #向下取整
    print(num_steps)
    def animate(num):
        step = (num*skip) % num_steps #%取余运算
        traj = 0

        #bb_min = gs[1].x[:, 0:2].min() # first two columns are velocity
        #bb_max = gs[1].x[:, 0:2].max() # use max and min velocity of gs dataset at the first step for both
                                          # gs and prediction plots
        #bb_min_evl = evl[1].x[:, 0:2].min()  # first two columns are velocity
        #bb_max_evl = evl[1].x[:, 0:2].max()  # use max and min velocity of gs dataset at the first step for both
                                          # gs and prediction plots
        count = 0

        for ax in axes:
            ax.cla()
            ax.set_aspect('equal')
            ax.set_axis_off()

            pos = gs[step].mesh_pos
            faces = gs[step].cells
            if (count == 0):
                # ground truth
                
                temperature = gs[step].x[:, 0]
                
                title = 'Ground truth:'
            elif (count == 1):
                temperature = pred[step].x[:, 0]
                title = 'Prediction:'

            else:
                temperature= evl[step].x[:, 0]
                title = 'Prediction error:'
            
            triang = mtri.Triangulation(pos[:, 0], pos[:, 1], faces)
            
            
            '''if (count <= 1):
                # absolute values

                mesh_plot = ax.tripcolor(triang, temperature[:], cmap='jet', vmin=-0.5, vmax=0.5,  shading='flat' ) # x-velocity
                ax.triplot(triang, 'ko-', ms=0.5, lw=0.3)
            else:
                # error: (pred - gs)/gs
                mesh_plot = ax.tripcolor(triang, temperature[:], cmap='jet',vmin=-0.5, vmax=0.5, shading='flat' ) # x-velocity
                ax.triplot(triang, 'ko-', ms=0.5, lw=0.3)
                #ax.triplot(triang, lw=0.5, color='0.5')'''
            if (count <= 1):
                # absolute values

                mesh_plot = ax.tripcolor(triang, temperature[:], cmap='jet', vmin=0, vmax=1,  shading='flat' ) # x-velocity
                ax.triplot(triang, 'ko-', ms=0.5, lw=0.3)
            else:
                # error: (pred - gs)/gs
                mesh_plot = ax.tripcolor(triang, temperature[:], cmap='jet',vmin=-0.5, vmax=0.5, shading='flat' ) # x-velocity
                ax.triplot(triang, 'ko-', ms=0.5, lw=0.3)
                #ax.triplot(triang, lw=0.5, color='0.5')

            ax.set_title('{} Trajectory {} Step {}'.format(title, traj, step), fontsize = '20')
            #ax.color

            #if (count == 0):
            divider = make_axes_locatable(ax)#在ax上创建一个可分离区域
            cax = divider.append_axes('right', size='5%', pad=0.05)
            clb = fig.colorbar(mesh_plot, cax=cax, orientation='vertical')
            clb.ax.tick_params(labelsize=20)

            '''clb.ax.set_title('temperature ',
                             fontdict = {'fontsize': 20})'''
            clb.ax.set_title('x velocity  (m/s)',
                             fontdict = {'fontsize': 20})
            count += 1
        return fig,

    # Save animation for visualization
    if not os.path.exists(path):
        os.makedirs(path)

    if (save_anim):
        gs_anim = animation.FuncAnimation(fig, animate, frames=num_frames, interval=1000)
        writergif = animation.PillowWriter(fps=10)

        anim_path = os.path.join(path, '{}_anim.gif'.format(name))
        gs_anim.save( anim_path, writer=writergif)
        plt.show(block=True)
    else:
        pass
def make_pred_animation(gs, pred, path, name , step):
    '''
    input gs is a dataloader and each entry contains attributes of many timesteps.

    '''
    print('Generating pred velocity fields ...')
    fig, axes = plt.subplots(2, 1, figsize=(20, 10))
    traj = 0
    count = 0

    for ax in axes:
        ax.cla()
        ax.set_aspect('equal')
        ax.set_axis_off()

        pos = gs.mesh_pos
        faces = gs.cells
        if (count == 0):
                # ground truth
                velocity = gs.x[:, 0:2]
                title = 'Ground truth:'
        else :
                velocity = pred.x[:, 0:2]
                title = 'Prediction:'
        

           

        triang = mtri.Triangulation(pos[:, 0], pos[:, 1], faces)
        if (count <= 1):
                # absolute values

                mesh_plot = ax.tripcolor(triang, velocity[:, 0], cmap='jet',vmin= 0, vmax=1,  shading='gouraud' ) # x-velocity
                ax.triplot(triang, 'ko-', ms=0.5, lw=0.3)
            

        ax.set_title('{} Trajectory {} Step {}'.format(title, traj, step), fontsize = '20')
            #ax.color

            #if (count == 0):
        divider = make_axes_locatable(ax)#在ax上创建一个可分离区域
        cax = divider.append_axes('right', size='5%', pad=0.05)
        clb = fig.colorbar(mesh_plot, cax=cax, orientation='vertical')
        clb.ax.tick_params(labelsize=20)

        clb.ax.set_title('x velocity  (m/s)',
                             fontdict = {'fontsize': 20})
        count += 1
    

    # Save animation for visualization
    if not os.path.exists(path):
        os.makedirs(path)
    pred_anim_path = os.path.join(path, '{}_anim{}_Step.pdf'.format(name,step))
    
    fig.savefig(pred_anim_path)






# draw the plot for point over time series
def point_over_time(ori_data, edg_data,j,ts , fieldName,path):
        point = [0,ts]
       
        x = np.linspace(point[0],int(point[1]-1),int(point[1]-point[0]))
        y_u=[]
        y_1_u=[]
        for i in range(ts):
            y_u.append(ori_data[i].x[j,0])
            y_1_u.append(edg_data[i].x[j,0])
        print(len(y_u))
        f = plt.figure()
        plt.plot(x, y_u,'k',  linewidth = 0.7)
        plt.plot(x, y_1_u, 'r--', linewidth = 0.7)
        plt.xlim((point[0], point[1]))# range
        plt.ylim((0, 5))

        plt.title(fieldName + ' Magnitude')
        plt.xlabel('Time(s)')
        plt.ylabel(fieldName)
        
        plt.legend(['Full Model', 'GNN ROM '], loc='lower right')
        #f.savefig('/root/data1/EMD/'+str(j)+'velocity.pdf', bbox_inches='tight')
        f.savefig(path, bbox_inches='tight')
def pearson_value(ori_data, rom_data):
    # print(ori_data.shape)
    # print(len(ori_data))
    pearson_value = []
    if len(ori_data) != len(rom_data):
        print('the length of these two array do not match')
    else:
        for i in range(len(rom_data)):
            row_1 = np.reshape(ori_data[i],(-1,1))
            row_2 = np.reshape(rom_data[i],(-1,1))
            data = np.hstack((row_1,row_2))
            df = pd.DataFrame(data=data[0:,0:],columns=['11','22'])
            pearson = df.corr() # pearson cc  # 
            # pearson = data.corr('spearman') # spearman cc  
            pear_value=pearson.iloc[0:1,1:2]
            value = pear_value.values
            if i == 0:
                pearson_value = value
            else:
                pearson_value = np.hstack((pearson_value,value)) 
        pearson_value = np.reshape(pearson_value,(-1,1))
    return np.array(pearson_value)

def pcc_of_two(ori_data, rom_data):

    if ori_data.ndim == rom_data.ndim:
        if rom_data.ndim == 3:
            y_u = ori_data[...,0] # u
            y_v = ori_data[...,1] # v
            y_0_u = rom_data[...,0] # u
            y_0_v = rom_data[...,1] # v
            pcc_x = pearson_value(y_u, y_0_u)
            pcc_y = pearson_value(y_v, y_0_v)
            plt.figure(1)
            plt.plot(pcc_x)
            plt.xlabel('Time(s)',{'size' : 11})
            plt.ylabel('Pearson Correlation Coefficient of x axis',{'size' : 11})
            plt.figure(2)
            plt.plot(pcc_y)
            plt.xlabel('Time(s)',{'size' : 11})
            plt.ylabel('Pearson Correlation Coefficient of y axis',{'size' : 11})
            plt.show()
        elif rom_data.ndim == 2:
            pcc = pearson_value(ori_data, rom_data)
            # print(pcc.shape)
            plt.figure(1)
            # x = np.linspace(0,pcc.shape[0], num = pcc.shape[0])
            plt.plot(pcc)
            plt.ylim((0, 1))
            plt.xlabel('Time(s)',{'size' : 11})
            plt.ylabel('Pearson Correlation Coefficient',{'size' : 11})
            plt.show()
    else:
        print('the dimension of these two series are not equal. Please check them.')
def mse(ori_data, rom_data):

    rmse_value = []
    if len(ori_data) != len(rom_data):
        print('the length of these two array do not match')
    else:
        for i in range(len(rom_data)):
            value = mean_squared_error(ori_data[i], rom_data[i])
            if i == 0:
                rmse_value = value
            else:
                rmse_value = np.hstack((rmse_value,value))
        rmse_value = np.reshape(rmse_value,(-1,1))
    return rmse_value

def mse_of_two(ori_data, rom_data):
    # , rom_data_1):
    if ori_data.ndim == rom_data.ndim:
        if rom_data.ndim == 3:
            y_u = ori_data[...,0] # u
            y_v = ori_data[...,1] # v
            y_0_u = rom_data[...,0] # u
            y_0_v = rom_data[...,1] # v
            rmse_x = mse(y_u, y_0_u)
            rmse_y = mse(y_v, y_0_v)
            plt.figure(1)
            plt.plot(rmse_x)
            plt.xlabel('Time(s)',{'size' : 11})
            plt.ylabel('RMSE of x axis',{'size' : 11})
            plt.figure(2)
            plt.plot(rmse_y)
            plt.xlabel('Time(s)',{'size' : 11})
            plt.ylabel('RMSE of y axis',{'size' : 11})
            plt.show()
        elif rom_data.ndim == 2:
            rmse_value = mse(ori_data, rom_data)
            # rmse_value_1 = rmse(ori_data, rom_data_1)
            # print(pcc.shape[0], pcc.shape[1])
            plt.figure(1)
            # x = np.linspace(0,rmse_value.shape[0], rmse_value.shape[0])
            # plt.plot(x, rmse_value, x, rmse_value_1)
            plt.plot(rmse_value)
            plt.ylim((0, 0.3))
            plt.xlabel('Time(s)',{'size' : 11})
            plt.ylabel('MSE',{'size' : 11})
            # plt.legend(['7', '8'], loc='lower right')   
            plt.show()
    else:
        print('the dimension of these two series are not equal. Please check them.')

def dataset_cat(datalist,n):
      
      x=torch.cat((datalist[0].x,datalist[1].x,datalist[2].x,datalist[3].x),dim=0)
      
      edge_index=torch.cat((datalist[0].edge_index,datalist[1].edge_index,
                      datalist[2].edge_index,datalist[3].edge_index),dim=1)
      
      edge_attr=torch.cat((datalist[0].edge_attr,datalist[1].edge_attr,
                      datalist[2].edge_attr,datalist[3].edge_attr),dim=0)
      
      if n==3:
            pos=datalist[0].mesh_pos
            data=Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
      if n==4:
            pos=torch.cat((datalist[0].mesh_pos,datalist[1].mesh_pos,
                      datalist[2].mesh_pos,datalist[3].mesh_pos),dim=0)
            data=Data(x=x, edge_index=edge_index, edge_attr=edge_attr,mesh_pos=pos)
      return pos,data


def transform_vector(data, satart_number_ts,number_ts, originalFolder, destinationFolder, fileName):

    folder = os.path.exists(destinationFolder)

    if not folder: 
        print('start to create the destination folder')   
        os.makedirs(destinationFolder)       
        copyFiles(originalFolder,destinationFolder) 

    print('start to store data as a new variable')
    
    for i in range(number_ts):
        f_filename = destinationFolder + fileName + str(i+satart_number_ts)+ ".vtk"
        reader=vtk.vtkPolyDataReader()
        reader.SetFileName(f_filename)  # SetFileName设置要读取的vtk文件
        reader.ReadAllScalarsOn()
        reader.ReadAllVectorsOn()
        reader.ReadAllTensorsOn()
        reader.Update()

        vtkdata=reader.GetOutput()  # GetOutput获取文件的数据
        num_points=vtkdata.GetNumberOfPoints() 
        point_data =vtkdata.GetPointData()
        a=data[i].x
        a=a.numpy()
    
        velocity_array = point_data.GetArray('U')
        #print(velocity_array.GetTuple(0)[0:2])
        for j in range(num_points):
            original_velocity = list(velocity_array.GetTuple(j))
            original_velocity[0:2]=tuple(a[j])
            velocity_array.SetTuple(j, original_velocity)

        writer = vtk.vtkGenericDataObjectWriter()
        writer.SetFileName(f_filename)
        writer.SetInputData(vtkdata)
        writer.Write()

    print('transform succeed')	

def copyFiles(sourceDir,targetDir):
    if sourceDir.find("exceptionfolder")>0:
        return

    for file in os.listdir(sourceDir):
        sourceFile = os.path.join(sourceDir,file)
        targetFile = os.path.join(targetDir,file)

        if os.path.isfile(sourceFile):
            if not os.path.exists(targetDir):
                os.makedirs(targetDir)
            if not os.path.exists(targetFile) or (os.path.exists(targetFile) and (os.path.getsize(targetFile) !=os.path.getsize(sourceFile))):
                open(targetFile, "wb").write(open(sourceFile, "rb").read())
                # print(targetFile+" copy succeeded")

        if os.path.isdir(sourceFile):
            copyFiles(sourceFile, targetFile)       

'''def mp_ae_dataprocess(x):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    y=torch.zeros(1,x.shape[0]*x.shape[1]).to(device)
    for i in range(x.shape[1]):
        y[0,i*x.shape[0]:(i+1)*x.shape[0]]=x[:,i]
    
    return y
    
def mp_ae_dataprocess_inv(x,a,b):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    y=torch.zeros(a,b).to(device)
    for i in range(b):
        y[:,i]=x[0,i*a:(i+1)*a]
    return y'''
def mp_ae_dataprocess(x):
    device = 'cpu'
    y=torch.zeros(1,x.shape[0]*x.shape[1]).to(device)
    for i in range(x.shape[1]):
        y[0,i*x.shape[0]:(i+1)*x.shape[0]]=x[:,i]
    
    return y
    
def mp_ae_dataprocess_inv(x,a,b):
    device = 'cpu'
    y=torch.zeros(a,b).to(device)
    for i in range(b):
        y[:,i]=x[0,i*a:(i+1)*a]
    return y


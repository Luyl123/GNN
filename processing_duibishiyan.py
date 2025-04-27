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
import math
from keras import backend as K
def test(model, loader, criterion,per_t,batch_size,EDmean_vec_x,EDstd_vec_x,EDmean_vec_edge,EDstd_vec_edge):
    model.eval()
    mse_total = 0.0
    count = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    
    with torch.no_grad():
        for data in loader:
            data=data.to(device)
            out,  _ = model(data, per_t , batch_size,EDmean_vec_x,EDstd_vec_x,EDmean_vec_edge,EDstd_vec_edge,batch=None, return_mask=False)
            data.x=(data.x-EDmean_vec_x)/EDstd_vec_x
            mse = criterion(out, data.x)
            mse_total += mse.detach().item()
            count += 1
        mse_total = mse_total / count
    return mse_total

def model_1_train(N_epochs, model, train_loader, test_loader,per_t,batch_size,EDmean_vec_x,EDstd_vec_x,EDmean_vec_edge,EDstd_vec_edge):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                patience=3, threshold=0.0000001, threshold_mode='rel',
                                cooldown=0, min_lr=1e-7, eps=1e-08, verbose=True)  #默认使用损失值监控
    criterion = torch.nn.MSELoss()
    
    acc=torch.nn.MSELoss(reduction='sum')
    train_hist = np.zeros(N_epochs)
    test_hist = np.zeros(N_epochs)
    update_iter = 0
    #per_t=5
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    
    for epoch in trange(N_epochs, desc="Training", unit="Epochs"):
    
        time_epoch = time.time()
        model.train()
        batch_count = 0
        train_mse = 0
        train_RMSE=0
        train_acc=0


        for step, data in enumerate(train_loader):
            data=data.to(device)

            out, _ = model(data, per_t , batch_size,EDmean_vec_x,EDstd_vec_x,EDmean_vec_edge,EDstd_vec_edge,batch=None, return_mask=False)
            data.x=(data.x-EDmean_vec_x)/EDstd_vec_x
            loss = criterion(out, data.x)
            train_acc_=acc(out, data.x)
            loss.backward()
            optimizer.step()     #一个batch一更新
            optimizer.zero_grad()
            train_mse += loss.item()
            train_RMSE+=math.sqrt(loss.item())
            #print(torch.sum(data.x))
            train_acc+=(math.sqrt(train_acc_.item())/torch.norm(data.x, p=2))
            batch_count += 1
            update_iter += 1
           
            


       

        test_mse = test(model, test_loader, criterion, per_t,batch_size,EDmean_vec_x,EDstd_vec_x,EDmean_vec_edge,EDstd_vec_edge)

        time_epoch = time.time() - time_epoch
        train_mse = train_mse/batch_count   #一个epoch的平均mse
        train_RMSE=train_RMSE/batch_count
        train_acc=train_acc/batch_count

        #if(epoch%2==0):
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
    pathED = '/root/data1/duibishiyan/model_1_ED_state_dict.pt'    
    #pathED = '/root/data1/EMD/MP_ED_state_dict.pt'   #没加入全局信
    torch.save(model.state_dict(), pathED)

    return train_hist, test_hist


def transform_vector(data, satart_number_ts,number_ts, originalFolder, destinationFolder, fileName):

    folder = os.path.exists(destinationFolder)

    if not folder: 
        print('start to create the destination folder')   
        os.makedirs(destinationFolder)       
        copyFiles(originalFolder,destinationFolder) 

    print('start to store data as a new variable')
    
    for i in range(number_ts):
        f_filename = destinationFolder + fileName + str(i+satart_number_ts)+ ".vtu"
        mesh =meshio.read(f_filename)
        a=data[i].x
        a=a.numpy()
        #a=a.reshape(a.shape[0])
        #print(mesh.point_data["Velocity"][45:55])
        #print(a[45:55])
        #mesh.point_data["Temperature"]=a
        mesh.point_data["Velocity"][:,0:2]=a
        
        #print(mesh.point_data["Velocity"][45:55])
        meshio.write(f_filename, mesh)
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

def model_2_train(N_epochs, model, train_loader, test_loader,per_t,batch_size,EDmean_vec_x,EDstd_vec_x,EDmean_vec_edge,EDstd_vec_edge):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                patience=3, threshold=0.0000001, threshold_mode='rel',
                                cooldown=0, min_lr=1e-7, eps=1e-08, verbose=True)  #默认使用损失值监控
    criterion = torch.nn.MSELoss()
    
    acc=torch.nn.MSELoss(reduction='sum')
    train_hist = np.zeros(N_epochs)
    test_hist = np.zeros(N_epochs)
    update_iter = 0
    #per_t=5
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    
    for epoch in trange(N_epochs, desc="Training", unit="Epochs"):
    
        time_epoch = time.time()
        model.train()
        batch_count = 0
        train_mse = 0
        train_RMSE=0
        train_acc=0


        for step, data in enumerate(train_loader):
            data=data.to(device)

            out, _ = model(data, per_t , batch_size,EDmean_vec_x,EDstd_vec_x,EDmean_vec_edge,EDstd_vec_edge,batch=None, return_mask=False)
            data.x=(data.x-EDmean_vec_x)/EDstd_vec_x
            loss = criterion(out, data.x)
            train_acc_=acc(out, data.x)
            loss.backward()
            optimizer.step()     #一个batch一更新
            optimizer.zero_grad()
            train_mse += loss.item()
            train_RMSE+=math.sqrt(loss.item())
            #print(torch.sum(data.x))
            train_acc+=(math.sqrt(train_acc_.item())/torch.norm(data.x, p=2))
            batch_count += 1
            update_iter += 1
           
            


       

        test_mse = test(model, test_loader, criterion, per_t,batch_size,EDmean_vec_x,EDstd_vec_x,EDmean_vec_edge,EDstd_vec_edge)

        time_epoch = time.time() - time_epoch
        train_mse = train_mse/batch_count   #一个epoch的平均mse
        train_RMSE=train_RMSE/batch_count
        train_acc=train_acc/batch_count

        #if(epoch%2==0):
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
    pathED = '/root/data1/duibishiyan/model_2_ED_state_dict.pt'    
    #pathED = '/root/data1/EMD/MP_ED_state_dict.pt'   #没加入全局信
    torch.save(model.state_dict(), pathED)

    return train_hist, test_hist 

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


def cc(ori_data, rom_data_0, rom_data_1, rom_data_2):
    pcc_0 = pearson_value(ori_data, rom_data_0)
    pcc_1 = pearson_value(ori_data, rom_data_1)
    pcc_2 = pearson_value(ori_data, rom_data_2)
    fig, ax = plt.subplots()
    x = np.linspace(0+25,25+(0.1*pcc_0.shape[0]),pcc_0.shape[0])
    print(pcc_0.shape, pcc_1.shape)
    ax.set_prop_cycle(color = ['red','#6a89cc','#f6b93b'], linestyle = ['-', '-', '-'],linewidth=[2,2,2])
    y_0 = pcc_0
    y_1 = pcc_1
    y_2 =pcc_2
   
    ax.plot(x, y_0, x, y_1, x, y_2)
     
            # plt.xlim((-0.1, 200.1))# range
    plt.ylim((0.973, 1))
    #plt.xlabel('TimeSteps',{'size' : 11})
    plt.xlabel('Time(s)',{'size' : 11})
    plt.ylabel('Pearson Correlation Coefficient',{'size' : 11})
            # plt.xticks(np.arange(0,200.1,25))
            # plt.yticks(np.arange(0.97,1.0001,0.01))
            # plt.legend(['D4:AE+TF', 'D8:AE+TF', 'D12:AE+TF', 'D4:PCA+TF', 'D8:PCA+TF', 'D12:PCA+TF'], loc='lower right')   
    plt.legend(['GAG-AE', 'MG-AE', 'GU-AE'],loc='lower center',ncol=3)   
    #plt.axis([250,400,0.973, 1])
    #plt.xticks([250, 260, 280, 300, 320, 340, 360, 380, 400])

    plt.axis([25,40,0.973, 1])
    plt.xticks([25, 30, 35, 40])
    plt.show()
    fig.savefig('/root/data1/duibishiyan/'+'test-Pcc.pdf', bbox_inches='tight')
    return y_0,y_1,y_2

def root_mean_squared_error(true, pred):
    return K.mean(K.square(pred - true))
def rmse(ori_data, rom_data):

    rmse_value = []
    if len(ori_data) != len(rom_data):
        print('the length of these two array do not match')
    else:
        for i in range(len(rom_data)):
            value = np.sqrt(root_mean_squared_error(ori_data[i], rom_data[i]))
            if i == 0:
                rmse_value = value
            else:
                rmse_value = np.hstack((rmse_value,value))
        rmse_value = np.reshape(rmse_value,(-1,1))
    return rmse_value
def rmse_over_time(ori_data, rom_data_0, rom_data_1, rom_data_2):
    rmse_0 = rmse(ori_data, rom_data_0)
    rmse_1 = rmse(ori_data, rom_data_1)
    rmse_2 = rmse(ori_data, rom_data_2)
  
    # rmse_4 = rmse(ori_data, rom_data_4)
    # rmse_5 = rmse(ori_data, rom_data_5)
 
    # plt.figure(1)
    # x = np.linspace(5,8,600)

    fig, ax = plt.subplots()
    #x = np.linspace(250+0,250+rmse_0.shape[0],rmse_0.shape[0])
    x = np.linspace(25+0,25+(0.1*rmse_0.shape[0]),rmse_0.shape[0])
    ax.set_prop_cycle(color = ['red','#6a89cc','#f6b93b'], linestyle = ['-', '-', '-'],linewidth=[2,2,2])
    # x = np.linspace(0,15,2875)
    y_0 = rmse_0
    y_1 = rmse_1
    y_2 = rmse_2
    # y_4 = rmse_4[-1800:-1200,:]
    # y_5 = rmse_5[-1800:-1200,:]
    # plt.title('Correlation Coefficient')
    ax.plot(x, y_0, x, y_1, x, y_2)
    # , x, rmse_4, linewidth = 0.6)
    # plt.xlim((-0.1, 200.1))# range
    plt.ylim((0, 0.07))
    #plt.xlabel('Time steps',{'size' : 11})
    plt.xlabel('Time(s)',{'size' : 11})
    plt.ylabel('RMSE',{'size' : 11})
    # plt.xticks(np.arange(0,200.1,25))
    # plt.yticks(np.arange(0,0.081,0.02))
    plt.legend(['GAG-AE', 'MG-AE', 'GU-AE'],loc='lower center',ncol=3)   
    #plt.axis([250,400,0, 0.07])
    #plt.xticks([250, 260, 280, 300, 320, 340, 360, 380, 400])

    plt.axis([25,40,0, 0.07])
    plt.xticks([25, 30, 35, 40])

    plt.show()
    fig.savefig('/root/data1/duibishiyan/'+'test-RMSE.pdf', bbox_inches='tight')
    return y_0,y_1,y_2


def point_over_time(ori_data,rom_data_0, rom_data_1, rom_data_2 ,j,ts , fieldName):
        point = [25+0,25+(0.1*ts)]
        #x = np.linspace(point[0],int(point[1]-1),int(point[1]-point[0]))
        x = np.linspace(point[0],point[1]-0.1,int(ts))
       
        y_u=[]
        y_0_u=[]
        y_1_u=[]
        y_2_u=[]
        for i in range(ts):
            y_u.append(ori_data[i].x[j,0])
            y_0_u.append(rom_data_0[i].x[j,0])
            y_1_u.append(rom_data_1[i].x[j,0])
            y_2_u.append(rom_data_2[i].x[j,0])
        print(len(y_u))
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.set_prop_cycle(color = ['#2f3542','red','#6a89cc','#f6b93b'], linestyle = ['-','--', '--', '--'],linewidth=[4,2,2,2])
        
        ax.plot(x, y_u, x, y_0_u,x, y_1_u, x, y_2_u)

        plt.ylim((0.38, 0.78))
        #plt.xlabel('TimeSteps')
        plt.xlabel('Time(s)')
        plt.ylabel(fieldName)
        
        plt.legend(['Full Model','GAG-AE', 'MG-AE', 'GU-AE'], loc='lower center',ncol=4)
        #plt.axis([250,400,0.38, 0.78])
        #plt.xticks([250, 260, 280, 300, 320, 340, 360, 380, 400])
        plt.axis([25,40,0.38, 0.78])
        plt.xticks([25, 30, 35, 40])

        fig.savefig('/root/data1/duibishiyan/test'+str(j)+'velocity.pdf', bbox_inches='tight')
        #plt.axis([320,340,0.38, 0.78])
        #plt.xticks([320, 325,330,335,340])
        plt.axis([32,34,0.38, 0.78])
        plt.xticks([32, 33, 34])
        fig.savefig('/root/data1/duibishiyan/test'+str(j)+'320-340-velocity.pdf', bbox_inches='tight')
        



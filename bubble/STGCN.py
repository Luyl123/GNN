import tensorflow.compat.v1 as tf
from os.path import join as pjoin
from scipy.sparse.linalg import eigs
import numpy as np
import time
import torch
import os
tf.disable_eager_execution()
def z_score(x, mean, std):
  
 
    return (x - mean) / std
def z_inverse(x, mean, std):
    return x * std + mean

def MSE(v, v_):
    
    return torch.mean((v_ - v) ** 2).item()


def RMSE(v, v_):
    
    return torch.sqrt(torch.mean((v_ - v) ** 2)).item()


def MAE(v, v_):
    
    return torch.mean(torch.abs(v_ - v)).item()
class Dataset(object):
    def __init__(self, data, stats):
        self.__data = data
        self.mean = stats['mean']
        self.std = stats['std']

    def get_data(self, type):
        return self.__data[type]

    def get_stats(self):
        return {'mean': self.mean, 'std': self.std}

    def get_len(self, type):
        return len(self.__data[type])

    def z_inverse(self, type):
        return self.__data[type] * self.std + self.mean
def scaled_laplacian(W):
    
    # d ->  diagonal degree matrix
   
    d = torch.sum(W, dim=1)
    # L -> graph Laplacian
    L = -W
    #L[np.diag_indices_from(L)] = d
    L[torch.arange(L.size(0)), torch.arange(L.size(1))] = d
    d_sqrt_inv = torch.where(d > 0, 1.0 / torch.sqrt(d), torch.zeros_like(d))  # d^(-1/2)
    D_inv_sqrt = torch.diag(d_sqrt_inv) 
    L = D_inv_sqrt @ L @ D_inv_sqrt
    eigenvalues,_ = torch.linalg.eigh(L)
    lambda_max = eigenvalues.max()
    print(lambda_max)
    return 2 * L / lambda_max - torch.eye(W.shape[0], device=W.device)

def cheb_poly_approx(L, Ks, n):
    L0 = torch.eye(n, device=L.device)
    L1 = L.clone() 

    if Ks > 1:
        L_list = [L0, L1]
        for i in range(Ks - 2):
            Ln =  2*torch.matmul(L, L1) - L0
            
            L_list.append(Ln)
            L0, L1 = L1, Ln
        # L_lsit [Ks, n*n], Lk [n, Ks*n]
        return np.concatenate(L_list, axis=-1)
    elif Ks == 1:
        return L0
    else:
        raise ValueError(f'ERROR: the size of spatial kernel must be greater than 1, but received "{Ks}".')
def seq_gen( data_seq,  n_frame):
    """n_frame=11
    data_seq=Data(x=(200,n_route,C_0),edge_index,edge_attr)"""
   
    n_slot = len(data_seq) - n_frame + 1#200-11+1
    n_route=data_seq[0].x.shape[0]
    C_0=data_seq[0].x.shape[1]

    tmp_seq =torch.zeros((n_slot, n_frame, n_route, C_0))
    for j in range(n_slot):
            
            for i in range(n_frame):
                tmp_seq[j,i,:,:]=data_seq[j+i].x
    return tmp_seq

def data_gen(data, n_train, n_val):
    

    seq_train = data[:n_train,:,:,:]
    seq_val =data[n_train:n_train+n_val,:,:,:]
    

    # x_stats: dict, the stats for the train dataset, including the value of mean and standard deviation.
    x_stats = {'mean': torch.mean(seq_train), 'std': torch.std(seq_train)}

    # x_train, x_val, x_test: np.array, [sample_size, n_frame, n_route, channel_size].
    
    x_train = z_score(seq_train, x_stats['mean'], x_stats['std'])
    x_val = z_score(seq_val, x_stats['mean'], x_stats['std'])
    

    x_data = {'train': x_train, 'val': x_val}
    dataset = Dataset(x_data, x_stats)
    return dataset,x_stats

def gen_batch(inputs, batch_size, dynamic_batch=False, shuffle=False):
    """batch_size=10"""
    len_inputs = inputs.shape[0]#140

    if shuffle:
        idx = np.arange(len_inputs)
        np.random.shuffle(idx)

    for start_idx in range(0, len_inputs, batch_size):
        end_idx = start_idx + batch_size
        if end_idx > len_inputs:
            if dynamic_batch:
                end_idx = len_inputs
            else:
                break
        if shuffle:
            slide = idx[start_idx:end_idx]
        else:
            slide = slice(start_idx, end_idx)
           
        

        yield inputs[slide]


def gconv(x, theta, Ks, c_in, c_out):
    
    
    # graph kernel: tensor, [n_route, Ks*n_route]
    
    kernel = tf.get_collection('graph_kernel')[0]
    
    
    n = kernel.shape[0]
    
    # x -> [batch_size, c_in, n_route] -> [batch_size*c_in, n_route]
    x_tmp = tf.reshape(tf.transpose(x, [0, 2, 1]), [-1, n])

    # x_mul = x_tmp * ker -> [batch_size*c_in, Ks*n_route] -> [batch_size, c_in, Ks, n_route]
    x_mul = tf.reshape(tf.matmul(x_tmp, kernel), [-1, c_in, Ks, n])
   
    # x_ker -> [batch_size, n_route, c_in, K_s] -> [batch_size*n_route, c_in*Ks]
    x_ker = tf.reshape(tf.transpose(x_mul, [0, 3, 1, 2]), [-1, c_in * Ks])
    
    # x_gconv -> [batch_size*n_route, c_out] -> [batch_size, n_route, c_out]
    x_gconv = tf.reshape(tf.matmul(x_ker, theta), [-1, n, c_out])
    
    
    
    return x_gconv


def layer_norm(x, scope):
   
    _, _, N, C = x.get_shape().as_list()
    mu, sigma = tf.nn.moments(x, axes=[2, 3], keep_dims=True)

    with tf.variable_scope(scope):
        gamma = tf.get_variable('gamma', initializer=tf.ones([1, 1, N, C]))
        beta = tf.get_variable('beta', initializer=tf.zeros([1, 1, N, C]))
        _x = (x - mu) / tf.sqrt(sigma + 1e-6) * gamma + beta
    return _x


def temporal_conv_layer(x, Kt, c_in, c_out, act_func='relu'):
    
    _, T, n, _ = x.get_shape().as_list()

    if c_in > c_out:
        w_input = tf.get_variable('wt_input', shape=[1, 1, c_in, c_out], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w_input))
        x_input = tf.nn.conv2d(x, w_input, strides=[1, 1, 1, 1], padding='SAME')
    elif c_in < c_out:
        # if the size of input channel is less than the output,
        # padding x to the same size of output channel.
        # Note, _.get_shape() cannot convert a partially known TensorShape to a Tensor.
        x_input = tf.concat([x, tf.zeros([tf.shape(x)[0], T, n, c_out - c_in])], axis=3)
    else:
        x_input = x

    # keep the original input for residual connection.
    x_input = x_input[:, Kt - 1:T, :, :]

    if act_func == 'GLU':
        # gated liner unit
        wt = tf.get_variable(name='wt', shape=[Kt, 1, c_in, 2 * c_out], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(wt))
        bt = tf.get_variable(name='bt', initializer=tf.zeros([2 * c_out]), dtype=tf.float32)
        x_conv = tf.nn.conv2d(x, wt, strides=[1, 1, 1, 1], padding='VALID') + bt
        return (x_conv[:, :, :, 0:c_out] + x_input) * tf.nn.sigmoid(x_conv[:, :, :, -c_out:])
    else:
        wt = tf.get_variable(name='wt', shape=[Kt, 1, c_in, c_out], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(wt))
        bt = tf.get_variable(name='bt', initializer=tf.zeros([c_out]), dtype=tf.float32)
        x_conv = tf.nn.conv2d(x, wt, strides=[1, 1, 1, 1], padding='VALID') + bt
        
        if act_func == 'linear':
            return x_conv
        elif act_func == 'sigmoid':
            return tf.nn.sigmoid(x_conv)
        elif act_func == 'relu':
            return tf.nn.relu(x_conv + x_input)
        else:
            raise ValueError(f'ERROR: activation function "{act_func}" is not defined.')


def spatio_conv_layer(x, Ks, c_in, c_out):
    
    _, T, n, _ = x.get_shape().as_list()

    if c_in > c_out:
        # bottleneck down-sampling
        w_input = tf.get_variable('ws_input', shape=[1, 1, c_in, c_out], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w_input))
        x_input = tf.nn.conv2d(x, w_input, strides=[1, 1, 1, 1], padding='SAME')
    elif c_in < c_out:
        # if the size of input channel is less than the output,
        # padding x to the same size of output channel.
        # Note, _.get_shape() cannot convert a partially known TensorShape to a Tensor.
        x_input = tf.concat([x, tf.zeros([tf.shape(x)[0], T, n, c_out - c_in])], axis=3)
    else:
        x_input = x

    ws = tf.get_variable(name='ws', shape=[Ks * c_in, c_out], dtype=tf.float32)
    
    tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(ws))
   
    
    
    bs = tf.get_variable(name='bs', initializer=tf.zeros([c_out]), dtype=tf.float32)
    
    
    # x -> [batch_size*time_step, n_route, c_in] -> [batch_size*time_step, n_route, c_out]
    x_gconv = gconv(tf.reshape(x, [-1, n, c_in]), ws, Ks, c_in, c_out) + bs
    # x_g -> [batch_size, time_step, n_route, c_out]
    x_gc = tf.reshape(x_gconv, [-1, T, n, c_out])
   
   
    return tf.nn.relu(x_gc[:, :, :, 0:c_out] + x_input)


def st_conv_block(x, Ks, Kt, channels, scope, keep_prob, act_func='GLU'):
    
    c_si, c_t, c_oo = channels

    with tf.variable_scope(f'stn_block_{scope}_in'):
        x_s = temporal_conv_layer(x, Kt, c_si, c_t, act_func=act_func)
        x_t = spatio_conv_layer(x_s, Ks, c_t, c_t)     
    with tf.variable_scope(f'stn_block_{scope}_out'):
        x_o = temporal_conv_layer(x_t, Kt, c_t, c_oo)
       
    x_ln = layer_norm(x_o, f'layer_norm_{scope}')
    return tf.nn.dropout(x_ln, keep_prob)

def fully_con_layer(x, n, channel, C0,scope):
    
    w = tf.get_variable(name=f'w_{scope}', shape=[1, 1, channel, C0], dtype=tf.float32)
    tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w))
    b = tf.get_variable(name=f'b_{scope}', initializer=tf.zeros([n, C0]), dtype=tf.float32)
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME') + b


def output_layer(x, T, scope, C0,act_func='GLU'):
    
    _, _, n, channel = x.get_shape().as_list()

    # maps multi-steps to one.
    with tf.variable_scope(f'{scope}_in'):
        x_i = temporal_conv_layer(x, T, channel, channel, act_func=act_func)
    x_ln = layer_norm(x_i, f'layer_norm_{scope}')
    with tf.variable_scope(f'{scope}_out'):
        x_o = temporal_conv_layer(x_ln, 1, channel, channel, act_func='sigmoid')
    # maps multi-channels to one.
    x_fc = fully_con_layer(x_o, n, channel, C0,scope)
    return x_fc




def build_model(inputs, n_his, Ks, Kt, blocks, C0,keep_prob):
    
    x = inputs[:, 0:n_his, :, :]

    # Ko>0: kernel size of temporal convolution in the output layer.
    Ko = n_his
    # ST-Block
    for i, channels in enumerate(blocks):
        x = st_conv_block(x, Ks, Kt, channels, i, keep_prob, act_func='GLU')
        Ko -= 2 * (Kt - 1)
    
    # Output Layer
    if Ko > 1:
        y = output_layer(x, Ko, 'output_layer',C0)
    else:
        raise ValueError(f'ERROR: kernel size Ko must be greater than 1, but received "{Ko}".')

    
    train_loss = tf.nn.l2_loss(y - inputs[:, n_his:n_his + 1, :, :])
    single_pred = y[:, 0, :, :]
    tf.add_to_collection(name='y_pred', value=single_pred)
    return train_loss, single_pred


def _pred(sess, y_pred, seq, batch_size, dynamic_batch=True):
    """n_pred=1"""
    pred_list = []
    for i in gen_batch(seq, min(batch_size, len(seq)), dynamic_batch=dynamic_batch):
        
        pred = sess.run(y_pred,
                            feed_dict={'data_input:0': i, 'keep_prob:0': 1.0})
        if isinstance(pred, list):
                pred = np.array(pred[0])#(batch,n,c)
        pred_list.append(pred)
    
    pred_array = np.concatenate(pred_list, axis=0)
    return pred_array
def evaluation(y, y_, x_stats):
    
    if isinstance(y_, np.ndarray):
        y_ = torch.tensor(y_, dtype=torch.float32)
   
    
    v = z_inverse(y, x_stats['mean'], x_stats['std'])
    v_ = z_inverse(y_, x_stats['mean'], x_stats['std'])
    return np.array([MSE(v, v_), MAE(v, v_), RMSE(v, v_)])
    


def model_inference(sess, pred, inputs, batch_size, n_his,x_stats):
    
    x_val = inputs.get_data('val')

    

    y_val= _pred(sess,pred,  x_val, batch_size, dynamic_batch=True)
    evl_val = evaluation(x_val[:, n_his, :, :], y_val, x_stats)

    return evl_val



def model_train(inputs, n, n_his, Ks, Kt,batch_size, epoch,opt,blocks, C0, lr,save_,x_stats):
    '''
    Train the base model.
    :param inputs: instance of class Dataset, data source for training.
    :param blocks: list, channel configs of st_conv blocks.
    :param args: instance of class argparse, args for training.
    n_his=10
    '''
    tf.disable_eager_execution()
    

    # Placeholder for model training
    x = tf.placeholder(tf.float32, [None, n_his + 1, n, C0], name='data_input')
   
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    
    # Define model loss
    train_loss, pred = build_model(x, n_his, Ks, Kt, blocks, C0,keep_prob)
    
    
    

    # Learning rate settings
    global_steps = tf.Variable(0, trainable=False)
    len_train = inputs.get_data('train').shape[0]
    
    if len_train % batch_size == 0:
        epoch_step = len_train / batch_size
    else:
        epoch_step = int(len_train / batch_size) + 1
    # Learning rate decay with rate 0.7 every 5 epochs.
    lr = tf.train.exponential_decay(lr, global_steps, decay_steps=5* epoch_step, decay_rate=0.5, staircase=True)
    
    step_op = tf.assign_add(global_steps, 1)
    
    with tf.control_dependencies([step_op]):
        if opt == 'RMSProp':
            train_op = tf.train.RMSPropOptimizer(lr).minimize(train_loss)
        elif opt == 'ADAM':
            train_op = tf.train.AdamOptimizer(lr).minimize(train_loss)
        else:
            raise ValueError(f'ERROR: optimizer "{opt}" is not defined.')

    
    saver = tf.train.Saver(max_to_keep=1)
    #model_path = "/root/data1/STGCN_result/saved_model/model.ckpt"
    model_path = "/root/data1/bubble_duibishiyan/saved_model/model.ckpt"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    

    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())

        for i in range(epoch):
            start_time = time.time()
            for j, x_batch in enumerate(
                    gen_batch(inputs.get_data('train'), batch_size, dynamic_batch=True, shuffle=False)):
                _ = sess.run([train_op], feed_dict={x: x_batch[:, 0:n_his + 1, :, :], keep_prob: 1.0})
                
                '''if i % 5 == 0 and j==(epoch_step-1):
                    loss_value = \
                        sess.run([train_loss],
                                 feed_dict={x:x_batch[:, 0:n_his + 1, :, :], keep_prob: 1.0})
                    print(f'Epoch {i:2d}, Step {j:3d}loss_value: [{loss_value[0]:.3f}],100Epoch Training Time {time.time() - start_time:.3f}s')'''
            

            
            evl_val = \
                model_inference(sess, pred, inputs, batch_size, n_his,x_stats)
            if i % 5 == 0:
                print(f'Epoch {i:2d}: '
                      f'MSE {evl_val[0]:7.3%}; '
                      f'MAE  {evl_val[1]:4.3f}; '
                      f'RMSE {evl_val[2]:6.3f}.')
                

            if (i + 1) % save_ == 0:
                
                saver.save(sess, model_path)
        
    print('Training model finished!')






def multi_pred(sess, y_pred, seq, n_his, n_pred):
    ''''seq=(1,11,n,c0), n_his=10, n_pred=190'''
    
    pred_list = []
    for i in gen_batch(seq,seq.shape[0]):
        test_seq = np.copy(seq)#seq(1,10,n,c0)
        for j in range(n_pred):
            pred = sess.run(y_pred,
                            feed_dict={'data_input:0': test_seq, 'keep_prob:0': 1.0})
            
            if isinstance(pred, list):
                pred = np.array(pred[0])
            
            test_seq[:, 0:n_his - 1, :, :] = test_seq[:, 1:n_his, :, :]
            test_seq[:, n_his - 1, :, :] = pred
            pred_list.append(pred)
    #  pred_array -> [1,n_pred, n_route, C_0)
    pred_array = np.concatenate(pred_list, axis=0)
    print(pred_array.shape)
    return pred_array#(190,n,c0)

def model_test(x_test, x_stats,  n_his, n_pred, load_path='/root/data1/bubble_duibishiyan/saved_model/'):
    
    
    model_path = tf.train.get_checkpoint_state(load_path).model_checkpoint_path#/root/data1/STGCN_result/saved_model/model.ckpt

    test_graph = tf.Graph()

    with test_graph.as_default():
        saver = tf.train.import_meta_graph(pjoin(f'{model_path}.meta'))

    with tf.Session(graph=test_graph) as test_sess:
        saver.restore(test_sess, tf.train.latest_checkpoint(load_path))#'/root/data1/STGCN_result/saved_model/model.ckpt'
        print(f'>> Loading saved model from {model_path} ...')

        pred = test_graph.get_collection('y_pred')


        y_test = multi_pred(test_sess, pred, x_test, n_his, n_pred)
        if isinstance(y_test, np.ndarray):
            y_test = torch.tensor(y_test, dtype=torch.float32)
        y_test = z_inverse( y_test, x_stats['mean'], x_stats['std'])

        
    print('Testing model finished!')
    return y_test

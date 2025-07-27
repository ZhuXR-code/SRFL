# torch.set_default_tensor_type(torch.DoubleTensor)

from options import args_parser,exp_details
from dataset import get_dataset
from models import MLP,CNNMnist,CNNFashion_Mnist,CNNCifar
from localupdate import LocalUpdate
from globalupdate import average_weights, test_inference

import collections
import copy
import logging
import math
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from numpy import transpose
from tqdm import tqdm

import numpy

# 导入 Pyfhel 库
from Pyfhel import Pyfhel, PyCtxt
from Pyfhel import Pyfhel, PyPtxt, PyCtxt

from dataset import get_dataset
from globalupdate import test_inference
from localupdate import LocalUpdate
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from options import args_parser, exp_details
import sys
torch.set_default_dtype(torch.float64)  # 双精度浮点数

# 获取当前文件名
current_file_name = os.path.basename(__file__)
# 配置日志记录器
logging.basicConfig(
    filename=f'{current_file_name}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

if __name__ == '__main__':
    start_time = time.time()

    args = args_parser()
    exp_details(args)

    # device = 'cuda' if args.gpu else 'cpu'
    device = 'cpu'

    train_dataset, test_dataset, user_samples = get_dataset(args)

    global_model = None
    if args.model == 'cnn':
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)
    elif args.model == 'mlp':
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
        global_model = MLP(dim_in=len_in,dim_hidden=64,
                           dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    global_model.to(device)
    global_model.train()

    global_weights = global_model.state_dict()
    global_weights_ = copy.deepcopy(global_weights)

    train_loss,train_accuracy = [],[]
    val_acc_list,net_list = [],[]
    cv_loss,cv_acc = [],[]
    val_loss_pre,counter = 0,0

    weights_numbers = copy.deepcopy(global_weights)
    
    
    # 初始化 Pyfhel 对象
    # https://pyfhel.readthedocs.io/en/latest/_autoexamples/Demo_5_CS_Client.html#sphx-glr-autoexamples-demo-5-cs-client-py
    HE = Pyfhel()  # Creating empty Pyfhel object

    # HE.contextGen(scheme='bgv', n=2**14, t_bits=20)  # Generate context for 'bfv'/'bgv'/'ckks' scheme

    #bgv_params = {
    #    'scheme': 'BGV',  # can also be 'bgv'
    #    'n': 2 ** 13,  # Polynomial modulus degree, the num. of slots per plaintext,
        #  of elements to be encoded in a single ciphertext in a
        #  2 by n/2 rectangular matrix (mind this shape for rotations!)
        #  Typ. 2^D for D in [10, 16]
    #    't': 65537,  # Plaintext modulus. Encrypted operations happen modulo t
        #  Must be prime such that t-1 be divisible by 2^N.
    #    't_bits': 20,  # Number of bits in t. Used to generate a suitable value
        #  for t. Overrides t if specified.
    #    'sec': 128,  # Security parameter. The equivalent length of AES key in bits.
        #  Sets the ciphertext modulus q, can be one of {128, 192, 256}
        #  More means more security but also slower computation.
    #}
    
    HE.contextGen(p=65537, base=2, intDigits=64, fracDigits = 32) 
    #HE.contextGen(**bgv_params)  # Generate context for bgv scheme
    HE.keyGen()  # Key Generation: generates a pair of public/secret keys
    #HE.rotateKeyGen()  # Rotate key generation --> Allows rotation/shifting
    #HE.relinKeyGen()  # Relinearization key generation



    for epoch in tqdm(list(range(args.epochs))):
        t11=time.time()
        


        local_weights, local_losses = [], []
        local_weights_ = []
        key_w=collections.OrderedDict()
        w_sum=[]
        index=0
        combined_tensor = None
        print(f'\n | Global Training Round : {epoch + 1} |\n')
        aux='nddfe'
        ctr=0
        Ct=[]
        
        value_shape=[]

        

        global_model.train()
        m = max(int(args.frac * args.num_users),1)
        index_of_users = np.random.choice(list(range(args.num_users)), m, replace=False)
        index = 0
        user_num=len(index_of_users)
        for k in index_of_users:
            value_length=[]
            w_value=[]
            combined_tensor = None

            local_model = LocalUpdate(args=args,dataset=train_dataset,
                                      index_of_samples=user_samples[k])

            t5=time.time()

            # w是局部模型，loss是模型的损失
            w,loss = local_model.local_train(model=copy.deepcopy(global_model),global_round=epoch)

            t6=time.time()
            print('train_time=',t6-t5)
            for key,value in w.items():
                value_length.append (value.numel())  #获取每个tensor的长度  
                #print(value)
                #print(len(value))
                #print(value.shape)
                key_w[key]=None #将原始数据的key赋值给新的字典key——w
                
            #print(key_w[key])
            #print(key_w)
            for key, value in w.items():
                #print(value)
                
                #print("original key is:",key)
                #print("original shape is:",value.shape)
                origin_shape = value.shape
                value_shape.append(origin_shape)#获取原始数据的维度
                v=value.reshape(-1)
                #print('v',v)
                if  combined_tensor is None:
                    combined_tensor = v
                else:
                    # 局部模型值拼接
                    combined_tensor=torch.cat((combined_tensor,v),dim=0)#多维tensor合并在一个里面
            
            #print("origin shape are:",value_shape)
            #print(combined_tensor,type(combined_tensor))   
            avlues_array = combined_tensor.numpy()#变为numpy数组
            #print('avlues_array',avlues_array,type(avlues_array))    
           # a=avlues_array.reshape(1,-1)
            #print('a',a)
           
            #print(w_value)    
            t1=time.time()

            # 加密
            ct1= [HE.encryptFrac(v) for v in avlues_array]  # 加密每个权重值

            print('size=',sys.getsizeof(ct1))
            t2=time.time()
            print('enc_time=',t2-t1)
            #local_weights_.append(copy.deepcopy(w))
            Ct.append(ct1)
            #print(Ct)
            #if args.model == 'cnn':
            #    for key,_ in list(w.items()):
                    
            #        N = w[key].numel()
            #        weights_numbers[key] = torch.tensor(N)
            #        M = max(int(args.compression_ratio * N), 1)

            #        w_dct = dct(w[key].numpy().reshape((-1, 1)))
            #        e = epoch
            #        if e >= int(N / M):
            #            e = e - int(N / M) * int(epoch / int(N / M))
            #        y = w_dct[e * M:min((e + 1) * M, N), :]

            #        epsilon_user = args.epsilon + np.zeros_like(y)

            #        min_weight = min(y)
            #        max_weight = max(y)
            #        center = (max_weight + min_weight) / 2
            #        radius = (max_weight - center) if (max_weight - center) != 0. else 1
            #        miu = y - center
            #        Pr = (np.exp(epsilon_user) - 1) / (2 * np.exp(epsilon_user))
            #        u = np.zeros_like(y)
            #        for i in range(len(y)):
            #            u[i, 0] = np.random.binomial(1, Pr[i, :])

            #        for i in range(len(y)):
            #            if u[i, 0] > 0:
            #                y[i, :] = center + miu[i, :] * ((np.exp(epsilon_user[i, :]) + 1) / (np.exp(epsilon_user[i, :]) - 1))
            #            else:
            #                y[i, :] = center + miu[i, :] * ((np.exp(epsilon_user[i, :]) - 1) / (np.exp(epsilon_user[i, :]) + 1))

                    #     if u[i, 0] > 0:
                    #         y[i, :] = center + radius * ((np.exp(epsilon_user) + 1) / (np.exp(epsilon_user) - 1))
                    #     else:
                    #         y[i, :] = center - radius * ((np.exp(epsilon_user) + 1) / (np.exp(epsilon_user) - 1))

            #        w[key] = torch.from_numpy(y)

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

            print(f'| client-{index + 1} : {k} finished!!! |')
            index += 1
            
        print(f'\n | Client Training End!!! | \n')
        transpose1 = []
        # 解密与聚合
        transpose_Ct = transpose(Ct).tolist()
        output = []
        t3 = time.time()
        for cipher in transpose_Ct:
            # 使用 Pyfhel 进行解密
            temp = numpy.sum(cipher)
            decrypted_values = HE.decryptFrac(temp) /(args.num_users)   #[HE.decryptBGV(c) for c in cipher]  # 解密每个密文
            output.append(decrypted_values)  # 将解密后的值添加到输出列表
        t4 = time.time()
        print('##### dec_time=', t4 - t3)
        logging.info(f'##### dec_time= {t4 - t3}')
        
        
            
        result3= [i/(user_num) for i in output]
        shapes_global = copy.deepcopy(global_weights)
        for key,_ in list(shapes_global.items()):
            shapes_global[key] = shapes_global[key].shape

        partial_global_weights_right = average_weights(local_weights)
        #print("right aggregation model", partial_global_weights_right)
        result_numpy=np.array(result3)#将一维列表转化为numpy数组
        result_tensor=torch.tensor(result_numpy)#将numpy数组转化为tensor
        #spilt_tensor=[]
        spilt_tensor1=[]
        start_index=0
        shape_index = 0
        tensor_index = 0
       # print(value_length)

        # 把output转回成模型的形式
        for length in value_length:#将tensor根据原始数组中的每个tensor长度分开成多个tensor
            #print("length is",length)
            
            spilt_tensor = result_numpy[start_index:start_index+length]
            start_index = start_index+length
            
            
            #print("value_shape is",value_shape[shape_index])
            spilt_tensor1.append (spilt_tensor.reshape(value_shape[shape_index]))#将tensor转为原始的维度
            shape_index = shape_index + 1
        #t1=dict(zip(key_w, spilt_tensor1))
        for key,tensor in key_w.items():#将tensor替换给原始数据中key对应的值
            
            partial_global_weights_right[key]=torch.from_numpy(spilt_tensor1[tensor_index])
            tensor_index =tensor_index+1
        partial_global_weights = partial_global_weights_right
        #print("our aggregation model", partial_global_weights_right)
        
        
        avg_loss = sum(local_losses) / len(local_losses)
        train_loss.append(avg_loss)

        #for key,_ in partial_global_weights.items():
         #   N = weights_numbers[key].item()
          #  M = max(int(args.compression_ratio * N), 1)
           # rec_matrix = np.zeros((N, 1))
            #e = epoch
            #if e >= int(N / M):
             #   e = e - int(N / M) * int(epoch / int(N / M))
            #rec_matrix[e * M:min((e + 1) * M, N), :] = partial_global_weights[key]
            #x_rec = idct(rec_matrix)
            #global_weights_1D = global_weights[key].numpy().reshape((-1, 1))
            #global_weights_1D[e * M:min((e + 1) * M, N), :] = (global_weights_1D[e * M:min((e + 1) * M, N), :] + x_rec[e * M:min((e + 1) * M, N), :]) / 2
            #global_weights[key] = torch.from_numpy(global_weights_1D.reshape(shapes_global[key]))

            # global_weights[key] = partial_global_weights[key].reshape(shapes_global[key])

            #print('key: ', key, '\t global_weights: ', global_weights[key].shape)

        global_model.load_state_dict(partial_global_weights)
        # global_model.load_state_dict(partial_global_weights)
        print(f'\n | Global Training Round : {epoch + 1} finished!!!!!!!!|\n')
        t12=time.time(

        )
        print('一轮训练时间',t12-t11)
        list_acc, list_loss = [], []
        global_model.eval()
        for k in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      index_of_samples=user_samples[k])
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc) / len(list_acc))

        print(f'\nAvg Training States after {epoch + 1} global rounds:')
        print(f'Avg Training Loss : {train_loss[-1]}')
        print('Avg Training Accuracy : {:.2f}% \n'.format(100 * train_accuracy[-1]))

        if math.isnan(train_loss[-1]):
            train_loss.pop()
            train_accuracy.pop()
            break

    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print(f'|---- Avg Train Loss: {np.mean(np.array(train_loss))}')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

    runtime = time.time() - start_time
    print(('\n Total Run Time: {0:0.4f}'.format(runtime)))

    
    data_log= {'Train Loss' : train_loss, 'Train Accuracy' : train_accuracy,
               'Test Loss' : test_loss, 'Test Accuracy' : test_acc}
    record = pd.DataFrame(data_log)
    record.to_csv('../log/MNIST/fed_{}_{}_E[{}]_iid[{}{}]_CR[{}]_epsilon[{}].csv'.
                format(args.dataset, args.model, args.epochs, args.iid, args.unequal, args.compression_ratio, args.epsilon))




    matplotlib.use('Agg')
    # # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_E[{}]_iid[{}]_CR[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs, args.iid, args.compression_ratio))

    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(list(range(len(train_accuracy))), train_accuracy)
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('../save/MNIST/fed_{}_{}_E[{}]_iid[{}{}]_CR[{}]_epsilon[{}].png'.
                format(args.dataset, args.model, args.epochs, args.iid, args.unequal, args.compression_ratio, args.epsilon))

import logging
import os
import time
from tqdm import tqdm
import copy
import math

import torch
# torch.set_default_tensor_type(torch.DoubleTensor)
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from utils.options import args_parser, exp_details
from utils.dataset import get_dataset
from utils.models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils.localupdate import LocalUpdate
from utils.globalupdate import average_weights, test_inference
from FE_inner_product_master.src.inner_product.single_input_fe.elgamal_ip import nddf
from numpy import transpose
import collections
import sys


# torch.set_default_dtype(torch.float64)
# torch.set_default_device('cpu')

def calculate_model_size(model):
    """
    计算模型参数大小（MB）
    参数：
        model : torch.nn.Module - 待计算的模型
    返回：
        total_size : float - 模型参数总大小（MB）
    """
    total_params = sum(p.numel() for p in model.parameters())
    element_size = model.parameters().__iter__().__next__().element_size()  # 获取参数元素大小
    total_size = total_params * element_size / (1024 * 1024)  # 转换为MB
    return total_size


def log_and_print(message):
    """
    日志记录工具函数（同时输出到控制台和日志文件）
    参数：
        message : str - 需要记录的信息
    """
    print(message)  # 打印到控制台
    logging.info(message)  # 记录到日志文件


if __name__ == '__main__':
    start_time = time.time()

    args = args_parser()
    exp_details(args)
    # 参数解析与日志配置
    current_file_name = os.path.basename(__file__)  # 获取当前文件名用于日志命名
    logging.basicConfig(
        filename=f'{current_file_name}.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
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
        global_model = MLP(dim_in=len_in, dim_hidden=64,
                           dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    global_model.to(device)
    global_model.train()

    global_weights = global_model.state_dict()
    global_weights_ = copy.deepcopy(global_weights)

    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0

    weights_numbers = copy.deepcopy(global_weights)

    for epoch in tqdm(list(range(args.epochs))):
        t11 = time.time()
        pk, sk = nddf.set_up(1024, 4)
        y = [1, 1]

        # 公私钥分配，三个角色公私钥不能有重合，是分开的
        pk0, sk0 = pk[0], sk[0]  # SEVER
        pk_i, sk_i = [pk[1], pk[2]], [sk[1], sk[2]]
        pk_3party, sk_3party = pk[3], sk[3]  # TP

        local_weights, local_losses = [], []
        local_weights_ = []
        key_w = collections.OrderedDict()
        w_sum = []
        index = 0
        combined_tensor = None
        log_and_print(f'\n | Global Training Round : {epoch + 1} |\n')
        aux = 'nddfe'
        ctr = 0
        Ct = []

        value_shape = []

        # 函数秘钥的生成
        skf = nddf.KeyDerive(sk_3party, pk_i, sk_i, ctr, y, aux)

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        index_of_users = np.random.choice(list(range(args.num_users)), m, replace=False)
        index = 0
        user_num = len(index_of_users)
        for k in index_of_users:
            value_length = []
            w_value = []
            combined_tensor = None

            local_model = LocalUpdate(args=args, dataset=train_dataset, index_of_samples=user_samples[k])

            t5 = time.time()

            # w是局部模型，loss是模型的损失
            w, loss = local_model.local_train(model=copy.deepcopy(global_model), global_round=epoch)

            t6 = time.time()
            log_and_print(f'train_time= {t6 - t5}')
            for key, value in w.items():
                value_length.append(value.numel())  # 获取每个tensor的长度
                key_w[key] = None  # 将原始数据的key赋值给新的字典key——w

            for key, value in w.items():
                log_and_print(f"original key is: {key}")
                log_and_print(f"original shape is: {value.shape}")
                origin_shape = value.shape
                value_shape.append(origin_shape)  # 获取原始数据的维度
                v = value.reshape(-1)
                if combined_tensor is None:
                    combined_tensor = v
                else:
                    # 局部模型值拼接
                    combined_tensor = torch.cat((combined_tensor, v), dim=0)  # 多维tensor合并在一个里面

            log_and_print(f"origin shape are: {value_shape}")
            avlues_array = combined_tensor.numpy()  # 变为numpy数组
            a_list = avlues_array.tolist()  # 转化为一维列表
            for i in range(len(a_list)):
                w_value.append(round((a_list[i] + 10) * 100000))
            t1 = time.time()

            # 加密
            ct1 = nddf.Encrypt(pk_3party, sk_i[index], pk0, ctr, w_value, aux)

            log_and_print(f'ct1 size= {sys.getsizeof(ct1)}')
            t2 = time.time()
            log_and_print(f'enc_time= {t2 - t1}')
            Ct.append(ct1)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

            log_and_print(f'| client-{index + 1} : {k} finished!!! |')
            index += 1

        log_and_print(f'\n | Client Training End!!! | \n')
        transpose1 = transpose(Ct).tolist()
        output = []
        t3 = time.time()

        for j in transpose1:
            output1 = nddf.Decrypt(pk_3party, skf, sk0, j, y)
            output.append(output1)

        result11 = []
        t4 = time.time()
        log_and_print(f'dec_time= {t4 - t3}')
        for i in output:
            result11.append(i / 100000 - 10 * user_num)

        result3 = [i / (user_num) for i in result11]
        log_and_print(f'len(result3)={len(result3)} | len(result11)={len(result11)}')
        shapes_global = copy.deepcopy(global_weights)
        for key, _ in list(shapes_global.items()):
            shapes_global[key] = shapes_global[key].shape

        partial_global_weights_right = average_weights(local_weights)
        # log_and_print(f"right aggregation model {partial_global_weights_right}")
        result_numpy = np.array(result3)
        result_tensor = torch.tensor(result_numpy)
        spilt_tensor1 = []
        start_index = 0
        shape_index = 0
        tensor_index = 0

        for length in value_length:
            log_and_print(f"length is {length}")

            spilt_tensor = result_numpy[start_index:start_index + length]
            start_index = start_index + length

            log_and_print(f"value_shape is {value_shape[shape_index]}")
            spilt_tensor1.append(spilt_tensor.reshape(value_shape[shape_index]))
            shape_index = shape_index + 1

        for key, tensor in key_w.items():
            partial_global_weights_right[key] = torch.from_numpy(spilt_tensor1[tensor_index])
            tensor_index = tensor_index + 1
        partial_global_weights = partial_global_weights_right
        log_and_print(f"our aggregation model {partial_global_weights_right}")

        avg_loss = sum(local_losses) / len(local_losses)
        train_loss.append(avg_loss)

        global_model.load_state_dict(partial_global_weights)
        log_and_print(f'\n | Global Training Round : {epoch + 1} finished!!!!!!!!|\n')
        t12 = time.time()
        log_and_print(f'一轮训练时间 {t12 - t11}')
        list_acc, list_loss = [], []
        global_model.eval()
        for k in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      index_of_samples=user_samples[k])
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc) / len(list_acc))

        log_and_print(f'\nAvg Training States after {epoch + 1} global rounds:')
        log_and_print(f'Avg Training Loss : {train_loss[-1]}')
        log_and_print('Avg Training Accuracy : {:.2f}% \n'.format(100 * train_accuracy[-1]))

        if math.isnan(train_loss[-1]):
            train_loss.pop()
            train_accuracy.pop()
            break

    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    log_and_print(f' \n Results after {args.epochs} global rounds of training:')
    log_and_print(f'|---- Avg Train Loss: {np.mean(np.array(train_loss))}')
    log_and_print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
    log_and_print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

    runtime = time.time() - start_time
    log_and_print(('\n Total Run Time: {0:0.4f}'.format(runtime)))

    data_log = {'Train Loss': train_loss, 'Train Accuracy': train_accuracy,
                'Test Loss': test_loss, 'Test Accuracy': test_acc}
    record = pd.DataFrame(data_log)
    record.to_csv('../log/MNIST/fed_{}_{}_E[{}]_iid[{}{}]_CR[{}]_epsilon[{}].csv'.
                  format(args.dataset, args.model, args.epochs, args.iid, args.unequal, args.compression_ratio,
                         args.epsilon))

    matplotlib.use('Agg')
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(list(range(len(train_accuracy))), train_accuracy)
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('../save/MNIST/fed_{}_{}_E[{}]_iid[{}{}]_CR[{}]_epsilon[{}].png'.
                format(args.dataset, args.model, args.epochs, args.iid, args.unequal, args.compression_ratio,
                       args.epsilon))

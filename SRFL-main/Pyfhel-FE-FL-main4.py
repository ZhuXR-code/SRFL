import collections
import copy
import logging
import math
import os
import time
from datetime import datetime
import numpy

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from numpy import transpose
from tqdm import tqdm

# 导入 Pyfhel 库
from Pyfhel import Pyfhel, PyCtxt
from Pyfhel import Pyfhel, PyPtxt, PyCtxt

from utils.dataset import get_dataset
from utils.globalupdate import test_inference
from utils.localupdate import LocalUpdate
from utils.models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils.options import args_parser, exp_details

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
    print(f'======start_time======: {datetime.now()}')
    logging.info(f'======start_time======: {datetime.now()}')

    # 数值缩放参数
    scale_factor = 10000  # 缩放因子
    offset = 10000        # 偏移量确保数值为正

    args = args_parser()
    exp_details(args)

    device = 'cuda' if args.gpu else 'cpu'

    # 数据加载与预处理
    train_dataset, test_dataset, user_samples = get_dataset(args)

    # 初始化全局模型
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
        len_in = np.prod(img_size)
        global_model = MLP(
            dim_in=len_in,
            dim_hidden=64,
            dim_out=args.num_classes
        )
    else:
        logging.error('\n Error: unrecognized model')
        exit('\n Error: unrecognized model')

    global_model.to(device)
    global_model.train()

    global_weights = global_model.state_dict()
    global_weights_ = copy.deepcopy(global_weights)

    # 训练指标
    train_loss, train_accuracy = [], []
    test_loss, test_acc = None, None

    # 初始化 Pyfhel 对象
    # https://pyfhel.readthedocs.io/en/latest/_autoexamples/Demo_5_CS_Client.html#sphx-glr-autoexamples-demo-5-cs-client-py
    HE = Pyfhel()  # Creating empty Pyfhel object

    # HE.contextGen(scheme='bgv', n=2**14, t_bits=20)  # Generate context for 'bfv'/'bgv'/'ckks' scheme

    bgv_params = {
        'scheme': 'CKKS',  # 使用 CKKS 方案
        'n': 2 ** 13,  # 多项式模数的度数
        't': 65537,  # 明文模数
        't_bits': 20,  # 明文模数的位数
        'sec': 128,  # 安全参数
        'qi_sizes': [60, 30, 30],  # 质数大小列表，用于定义密文模数
        'scale': 2 ** 30,  # 默认缩放因子，通常为 2 的幂
    }
    
    # HE.contextGen(p=65537, base=2, intDigits=64, fracDigits = 32)
    # ckks_params = {
    #     'scheme': 'CKKS',  # 使用 CKKS 方案
    #     'n': 2 ** 13,  # 多项式模数的度数，与旧版的 base 和 intDigits/fracDigits 相关
    #     't': 65537,  # 明文模数，与旧版的 p 相同
    #     # 'intDigits': 64,  # 整数部分的位数，与旧版的 intDigits 相同
    #     # 'fracDigits': 32,  # 小数部分的位数，与旧版的 fracDigits 相同
    #     'sec': 128  # 安全参数，设置为 128 位
    # }
    HE.contextGen(**bgv_params)  # Generate context for bgv scheme
    # HE.contextGen(ckks_params)  # Generate context for bgv scheme
    HE.keyGen()  # Key Generation: generates a pair of public/secret keys
    #HE.rotateKeyGen()  # Rotate key generation --> Allows rotation/shifting
    #HE.relinKeyGen()  # Relinearization key generation

    for epoch in tqdm(range(args.epochs)):
        round_start_time = time.time()
        print(f'\n | Global Training Round: {epoch + 1} |\n')
        logging.info(f'| Global Training Round: {epoch + 1} |')

        index = 0
        combined_tensor = None
        aux = 'Pyfhel'
        ctr = 0

        # 客户端选择
        m = max(int(args.frac * args.num_users), 1)
        selected_users = np.random.choice(
            range(args.num_users),
            m,
            replace=False
        )

        Ct = []  # 加密后的权重列表
        local_weights = []
        local_losses = []

        t_start = time.time()
        for idx, user_id in enumerate(selected_users):
            local_model = LocalUpdate(
                args=args,
                dataset=train_dataset,
                index_of_samples=user_samples[user_id]
            )

            t5 = time.time()
            # 本地训练
            w, loss = local_model.local_train(
                model=copy.deepcopy(global_model),
                global_round=epoch
            )

            t6 = time.time()
            print('train_time=', t6 - t5)
            logging.info(f'train_time= {t6 - t5}')

            # 处理 NaN 值
            for key, value in w.items():
                v = value.view(-1)
                if torch.isnan(v).any():
                    valid = v[~torch.isnan(v)]
                    mean_val = valid.mean().item() if valid.numel() else 0.0
                    v[torch.isnan(v)] = mean_val
                    logging.warning(
                        f"NaN detected in layer '{key}' of client {user_id}. "
                        f"Replaced with mean value: {mean_val}"
                    )
                w[key] = v.view(value.shape)

            # 权重拼接
            combined = None
            for key, value in w.items():
                flat = value.view(-1)
                combined = flat if combined is None else torch.cat([combined, flat])

            a_list = combined.numpy().tolist()

            # 对每个权重值进行缩放和加密
            w_value = [v * scale_factor + offset for v in a_list]
            # w_value = [int(round(v * scale_factor)) + offset for v in a_list]
            # w_value = np.array(
            #     [int(v * scale_factor) + offset for v in a_list],
            #     dtype=np.int64
            # )
            # print(w_value)

            # 使用 Pyfhel 进行加密
            t1 = time.time()
            #encrypted_values = [HE.encryptBGV(np.array([v], dtype=np.int64)) for v in w_value]  # 加密每个权重值
            # encrypted_values = [HE.encryptFrac(np.array([v], dtype=np.int64)) for v in w_value]  # 加密每个权重值
            encrypted_values = [HE.encodeFrac(np.array([v], dtype=np.float64)) for v in w_value]            # encrypted_values = HE.encryptBGV(w_value)  # 加密每个权重值
            # print(encrypted_values)
            
            t2 = time.time()
            print('enc_time=', t2 - t1)
            logging.info(f'enc_time= {t2 - t1}')

            Ct.append(encrypted_values)  # 将加密后的权重添加到列表中
            local_weights.append(w)
            local_losses.append(loss)

            print(f'| client-{index + 1} : {user_id} finished!!! |')
            logging.info(f'| client-{index + 1} : {user_id} finished!!! |')
            index += 1

        print(f'\n | Client Training End!!! | \n')
        logging.info(f'| Client Training End!!! |')

        # 解密与聚合
        transpose_Ct = transpose(Ct).tolist()
        output = []
        t3 = time.time()
        for cipher in transpose_Ct:
            # 使用 Pyfhel 进行解密
            temp = numpy.sum(cipher)
            decrypted_values = HE.decryptFrac(temp) /(args.num_users)   #[HE.decryptBGV(c) for c in cipher]  # 解密每个密文
            output.extend(decrypted_values)  # 将解密后的值添加到输出列表
        t4 = time.time()
        print('##### dec_time=', t4 - t3)
        logging.info(f'##### dec_time= {t4 - t3}')

        # 解密后的权重还原
        result = np.array(output, dtype=np.float128) / (scale_factor * len(selected_users))
        result = result.astype(np.float64)
        logging.info(f'result min: {np.min(result)}, result max: {np.max(result)}')

        result = result.flatten()

        # 权重分割与重塑
        key_w = collections.OrderedDict()
        value_length = []
        value_shape = []

        for key, value in global_weights.items():
            key_w[key] = None
            value_length.append(value.numel())
            value_shape.append(value.shape)

        spilt_tensors = []
        start = 0
        for length, shape in zip(value_length, value_shape):
            end = start + length
            tensor = result[start:end].reshape(shape)
            spilt_tensors.append(tensor)
            start = end

        new_weights = {
            key: torch.from_numpy(tensor)
            for key, tensor in zip(key_w.keys(), spilt_tensors)
        }

        # 更新全局模型
        global_model.load_state_dict(new_weights)
        avg_loss = sum(local_losses) / len(local_losses)
        train_loss.append(avg_loss)

        print(f'\n | Global Training Round : {epoch + 1} finished!!!!!!!!|\n')
        logging.info(f'| Global Training Round : {epoch + 1} finished!!!!!!!!|')
        t12 = time.time()

        print('一轮训练时间', t12 - t_start)
        logging.info(f'一轮训练时间 {t12 - t_start}')

        # 评估
        global_model.eval()
        acc_list = []
        for user_id in range(args.num_users):
            local_acc, _ = LocalUpdate(
                args=args,
                dataset=train_dataset,
                index_of_samples=user_samples[user_id]
            ).inference(global_model)
            acc_list.append(local_acc)
        train_accuracy.append(np.mean(acc_list))

        print(f'\nAvg Training States after {epoch+1} rounds:')
        print(f'Avg Training Loss: {avg_loss:.4f} | Avg Training Accuracy: {train_accuracy[-1]:.2%}')
        logging.info(
            f'Epoch {epoch+1} | Loss: {avg_loss:.4f} | Accuracy: {train_accuracy[-1]:.2%}'
        )

        if math.isnan(avg_loss):
            logging.error(f'NaN detected in loss at epoch {epoch+1}')
            break

    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    print(f' \n Results after {args.epochs} global rounds of training:')
    print(f'|---- Avg Train Loss: {np.mean(np.array(train_loss))}')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))
    logging.info(f'Results after {args.epochs} global rounds of training:')
    logging.info(f'|---- Avg Train Loss: {np.mean(np.array(train_loss))}')
    logging.info("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
    logging.info("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

    runtime = time.time() - start_time
    print(('\n Total Run Time: {0:0.4f}'.format(runtime)))
    logging.info(('Total Run Time: {0:0.4f}'.format(runtime)))

    data_log = {'Train Loss': train_loss, 'Train Accuracy': train_accuracy,
                'Test Loss': test_loss, 'Test Accuracy': test_acc}
    record = pd.DataFrame(data_log)
    log_dir = f'../log/{args.dataset}'
    os.makedirs(log_dir, exist_ok=True)
    df = pd.DataFrame({
        'Train Loss': train_loss,
        'Train Accuracy': train_accuracy,
        'Test Loss': test_loss,
        'Test Accuracy': test_acc
    })
    df.to_csv('{}/fed_{}_{}_E[{}]_iid[{}{}]_CR[{}]_epsilon[{}].csv'.
                  format(log_dir, args.dataset, args.model, args.epochs, args.iid, args.unequal, args.compression_ratio,
                         args.epsilon))

    matplotlib.use('Agg')
    plt.figure()
    plt.plot(train_accuracy, label='Training Accuracy')
    plt.title('Accuracy vs Communication Rounds')
    plt.xlabel('Rounds')
    plt.ylabel('Accuracy')
    plt.legend()
    save_dir = f'../save/{args.dataset}'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig('{}/fed_{}_{}_E[{}]_iid[{}{}]_CR[{}]_epsilon[{}].png'.
                format(save_dir, args.dataset, args.model, args.epochs, args.iid, args.unequal, args.compression_ratio,
                       args.epsilon))

    runtime = time.time() - start_time
    print(f'\nTotal Runtime: {runtime:.2f} seconds')
    logging.info(f'Total Runtime: {runtime:.2f} seconds')
    print(f'======end_time======: {datetime.now()}')
    logging.info(f'======end_time======: {datetime.now()}')

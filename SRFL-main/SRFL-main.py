# 导入必要的库
import multiprocessing
import os  # 操作系统接口模块，用于文件路径操作等
import time  # 时间模块，用于记录时间戳
from tqdm import tqdm  # 进度条工具，用于显示训练进度
import copy  # 深拷贝模块，用于复制对象
import math  # 数学模块，提供数学运算功能
import logging  # 日志记录模块，用于记录运行时信息
from decimal import Decimal, getcontext  # 高精度浮点数模块

import torch  # PyTorch深度学习框架
import numpy as np  # NumPy科学计算库
from scipy.fft import dct, idct  # 离散余弦变换及其逆变换
import matplotlib  # Matplotlib绘图库
import matplotlib.pyplot as plt  # Matplotlib绘图接口
import pandas as pd  # Pandas数据处理库

# 自定义模块导入
from utils.options import args_parser, exp_details  # 参数解析和实验细节打印
from utils.dataset import get_dataset  # 数据集加载函数
from utils.models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar  # 模型定义
from utils.localupdate import LocalUpdate  # 本地更新类，用于客户端训练
from utils.globalupdate import average_weights, test_inference, weighted_average_weights  # 全局模型更新和测试推理
from datetime import datetime  # 日期时间模块
from EDFE.src.inner_product.single_input_fe.elgamal_ip import edfe  # 功能加密模块
from numpy import transpose  # 转置操作
import collections  # 数据结构工具模块
import sys  # 系统模块，用于获取对象大小

# ====================== 全局配置 ======================
# 数值精度配置
# torch.set_default_dtype(torch.float64)  # 设置PyTorch默认数据类型为双精度浮点
getcontext().prec = 2000  # 设置Decimal模块计算精度（用于高精度加密计算）

# 通信配置
COMMUNICATION_SPEED_MBPS = 5  # 网络传输速率（单位：MB/s）
# 缩放因子
scale_factor = 100  # 用于调整数值的缩放比例
# 偏移量
offset = 10  # 确保计算结果非负的偏移值

# ====================== 工具函数 ======================
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
    log_and_print(f'model_size={total_params*element_size}')
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

# ====================== 主程序 ======================

if __name__ == '__main__':
    start_time = time.time()  # 记录程序开始时间

    # 参数解析与日志配置
    current_file_name = os.path.basename(__file__)  # 获取当前文件名用于日志命名
    logging.basicConfig(
        filename=f'{current_file_name}.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    args = args_parser()  # 解析命令行参数
    exp_details(args)  # 打印实验细节
    device = 'cpu'  # 使用CPU设备（原代码中注释掉了GPU选项）

    # 加载数据集，包括训练集、测试集和用户样本划分
    train_dataset, test_dataset, user_samples = get_dataset(args)

    global_model = None  # 初始化全局模型
    if args.model == 'cnn':  # 根据参数选择卷积神经网络模型
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)  # MNIST数据集对应的CNN模型
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)  # Fashion-MNIST数据集对应的CNN模型
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)  # CIFAR-10数据集对应的CNN模型
    elif args.model == 'mlp':  # 如果是多层感知机模型
        img_size = train_dataset[0][0].shape  # 获取图像尺寸
        len_in = 1
        for x in img_size:
            len_in *= x  # 计算输入维度
        global_model = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes)  # 定义MLP模型
    else:
        exit('Error: unrecognized model')  # 如果模型参数不合法，退出程序

    global_model.to(device)  # 将模型加载到指定设备
    global_model.train()  # 设置模型为训练模式

    global_weights = global_model.state_dict()  # 获取全局模型的权重
    global_weights_ = copy.deepcopy(global_weights)  # 深拷贝全局权重，用于后续操作

    # 初始化训练损失、准确率等变量
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0

    weights_numbers = copy.deepcopy(global_weights)  # 深拷贝全局权重，用于后续操作

    count_keys = args.num_users + 2  # 计算密钥数量（用户数+2）
    t0 = time.time()  # 记录密钥生成开始时间
    pk, sk, N = edfe.set_up(1024, count_keys)  # 生成公钥、私钥和模数N
    y = [1] * args.num_users  # 初始化y向量，用于内积计算

    # 分配公私钥，分为服务器、客户端和第三方
    pk0, sk0 = pk[0], sk[0]  # 服务器的公私钥
    pk_i, sk_i = pk[1:count_keys - 1], sk[1:count_keys - 1]  # 客户端的公私钥
    # pk_3party, sk_3party = pk[count_keys - 1], sk[count_keys - 1]  # 第三方的公私钥
    pk_3party, sk_3party = pk[-1], sk[-1]  # 第三方密钥

    t01 = time.time()  # 记录密钥生成结束时间
    log_and_print(f'keygen_time= {t01 - t0}')  # 打印并记录密钥生成时间

    upload_times, download_times = [], []  # 初始化上传和下载时间列表

    # 开始联邦学习训练循环
    for epoch in tqdm(list(range(args.epochs))):  # 遍历每个训练轮次
        t11 = time.time()  # 记录本轮训练开始时间
        round_start_time = time.time()
        log_and_print(f'++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')  # 打印并记录当前轮次
        log_and_print(f'++++++++++++++++++++++++| Global Training Round : {epoch + 1} |++++++++++++++++++++++++')  # 打印并记录当前轮次

        fe = edfe  # 引用功能加密模块
        local_weights, local_losses = [], []  # 初始化本地权重和损失列表
        local_weights_ = []  # 初始化本地权重备份列表
        key_w = collections.OrderedDict()  # 初始化有序字典，用于存储权重键值
        w_sum = []  # 初始化权重累加列表
        combined_tensor = None  # 初始化合并张量
        aux = 'edfe'  # 辅助字符串，用于功能加密
        ctr = 0  # 初始化计数器
        Ct = []  # 初始化密文列表
        value_shape = []  # 初始化形状列表

        skf_t1 = time.time()  # 记录功能密钥生成开始时间
        skf = edfe.KeyDerive(sk0, pk_i, ctr, y, aux)  # 生成功能密钥
        skf_t11 = time.time()  # 记录功能密钥生成结束时间
        log_and_print(f'functional keygen_time= {skf_t11 - skf_t1}')  # 打印并记录功能密钥生成时间

        global_model.train()  # 设置模型为训练模式

        m = max(int(args.frac * args.num_users), 1)  # 计算参与训练的客户端数量
        # 每轮选择客户端时
        index_of_users = np.random.choice(list(range(args.num_users)), m, replace=False)  # 不放回的随机选择客户端
        index = 0  # 初始化索引
        user_num = len(index_of_users)  # 获取实际参与训练的客户端数量

        value_length_list = []
        value_shape_list = []


        continue_flag = False
        # 遍历选中的客户端
        for idx_user, k in enumerate(index_of_users):
            value_length = []  # 初始化值长度列表
            w_value = []  # 初始化权重值列表
            combined_tensor = None  # 初始化合并张量

            # 创建本地更新对象前进行模型健康检查
            current_model = copy.deepcopy(global_model)

            # 检查模型参数是否包含无效值
            if any(torch.isnan(p).any() or not torch.isfinite(p).all() for p in current_model.parameters()):
                log_and_print(f"警告：客户端{k}在训练前检测到模型包含无效值（NaN/Inf）")
                continue_flag = True
                continue  # 跳过当前客户端训练

            # 创建本地更新对象
            local_model = LocalUpdate(args=args, dataset=train_dataset,index_of_samples=user_samples[k])
            t5 = time.time()  # 记录本地训练开始时间
            w, loss = local_model.local_train(model=copy.deepcopy(global_model), global_round=epoch)  # 本地训练


            # 添加权重有效性检查
            if any(torch.isnan(p).any() for p in w.values()):
                log_and_print(f"客户端{k}训练结果包含NaN，跳过该客户端")
                continue_flag = True
                continue  # 跳过后续处理

            t6 = time.time()  # 记录本地训练结束时间

            # 遍历本地模型权重
            for key, value in w.items():
                value_length.append(value.numel())  # 获取每个张量的元素数量
                key_w[key] = None  # 初始化权重字典

            # 权重处理
            for key, value in w.items():
                origin_shape = value.shape  # 获取原始形状
                value_shape.append(origin_shape)  # 记录形状
                v = value.reshape(-1)  # 将张量展平为一维
                if combined_tensor is None:
                    combined_tensor = v  # 初始化合并张量
                else:
                    combined_tensor = torch.cat((combined_tensor, v), dim=0)  # 合并张量


            # 记录当前值的长度到列表
            value_length_list.append(value_length)
            # 收集所有权重值的形状并记录到列表
            value_shape_list.append([value.shape for value in w.values()])

            log_and_print(f'| Train client-{idx_user + 1} : {k} loss={loss}   train_time= {t6 - t5}   | ')

            current_scale = scale_factor
            current_offset = offset

            w_value = [round((x.item() + offset) * scale_factor) for x in combined_tensor]
            t1 = time.time()  # 记录加密开始时间

            # ---------- 加密 ----------
            ct1 = fe.Encrypt(pk0, sk_i[index], ctr, w_value, aux, N)  # 加密权重

            t2 = time.time()  # 记录加密结束时间
            log_and_print(f'| Encrypt client-{idx_user + 1} : {k} enc_time= {t2 - t1} |  ct1 size= {sys.getsizeof(ct1)}')  # 打印并记录加密时间和密文大小
            Ct.append(ct1)  # 添加密文到列表
            local_weights.append(copy.deepcopy(w))  # 添加本地权重到列表
            local_losses.append(copy.deepcopy(loss))  # 添加本地损失到列表
            index += 1  # 更新索引


        log_and_print(f'| Client Training End!!! |')  # 打印并记录客户端训练结束信息

        '''解密'''
        transpose1 = transpose(Ct).tolist()  # 对密文矩阵进行转置
        output = []  # 初始化解密结果列表
        result11 = []  # 初始化解密结果列表
        t3 = time.time()  # 记录解密开始时间

        # 遍历转置后的密文矩阵
        for j in transpose1:
            output1 = fe.Decrypt(j, y, N)  # 解密 output1 是int
            output.append(output1)  # 添加解密结果到列表
            # 加密时候是 (x.item() + offset) * scale_factor，解密的时候由于是合并了的，所以要offset*user_num
            result11.append(output1/scale_factor - offset*user_num)

        log_and_print(f'len(output):{len(output)}   |   len(result11):{len(result11)}')
        t4 = time.time()  # 记录解密结束时间
        log_and_print(f'解密和反缩放时间 dec_time= {t4 - t3}')  # 打印并记录解密时间

        result3 = [i / (user_num) for i in result11]
        # 深拷贝全局权重并记录各参数张量的原始形状
        shapes_global = copy.deepcopy(global_weights)
        for key, _ in list(shapes_global.items()):
            shapes_global[key] = shapes_global[key].shape

        # 对本地模型权重进行加权平均（常规联邦平均）
        partial_global_weights_right = average_weights(local_weights)
        result_numpy = np.array(result3)  # 将一维列表转化为numpy数组
        result_tensor = torch.tensor(result_numpy)  # 将numpy数组转化为tensor
        # 初始化变量用于张量重构
        spilt_tensor1 = []
        start_index = 0
        tensor_index = 0

        # 把output转回成模型的形式，将扁平化的数值重构为原始张量形状
        log_and_print(f'value_length= {value_length}')
        for shape_index,length in enumerate(value_length):  # 将tensor根据原始数组中的每个tensor长度分开成多个tensor
            # 按原始参数长度切割numpy数组
            spilt_tensor = result_numpy[start_index:start_index + length]
            start_index = start_index + length
            log_and_print(f'shape_index={shape_index} | length= {length} | value_shape= {value_shape[shape_index]}')
            spilt_tensor1.append(spilt_tensor.reshape(value_shape[shape_index]))  # 将tensor转为原始的维度

        # 遍历键值对，将分割后的张量按顺序替换到对应键的权重中,同时维护张量索引计数器实现顺序替换
        for key, tensor in key_w.items():  # 将tensor替换给原始数据中key对应的值
            partial_global_weights_right[key] = torch.from_numpy(spilt_tensor1[tensor_index])
            tensor_index = tensor_index + 1
        # 更新全局权重引用并打印当前聚合模型参数
        partial_global_weights = partial_global_weights_right

        avg_loss = sum(local_losses) / len(local_losses)
        train_loss.append(avg_loss)

        # 更新全局模型
        global_model.load_state_dict(partial_global_weights)


        t12 = time.time()
        log_and_print(f'一轮训练时间 {t12 - t11}')
        # 计算准确率
        list_acc, list_loss = [], []
        global_model.eval()
        for k in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset, index_of_samples=user_samples[k])
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc) / len(list_acc))


        # 统计通信时间
        model_size_mb = calculate_model_size(global_model)
        upload_time = model_size_mb / COMMUNICATION_SPEED_MBPS
        download_time = model_size_mb / COMMUNICATION_SPEED_MBPS
        log_and_print(f'| 通信量: {model_size_mb:.9f}MB | 通信时间: Upload={upload_time:.2f}s | Download={download_time:.2f}s')
        log_and_print(f'Avg Training States after {epoch + 1} global rounds:')  # 打印并记录平均训练状态
        log_and_print(f'Avg Training Loss: {avg_loss:.4f}')
        log_and_print(f'Avg Training Accuracy: {train_accuracy[-1]:.2%}')


        if math.isnan(train_loss[-1]):  # 如果损失为NaN，终止训练
            train_loss.pop()
            train_accuracy.pop()
            break

    test_acc, test_loss = test_inference(args, global_model, test_dataset)  # 测试推理

    log_and_print(f' \n Results after {args.epochs} global rounds of training:')  # 打印并记录训练结果
    log_and_print(f'|---- Avg Train Loss: {np.mean(np.array(train_loss))}')  # 打印并记录平均训练损失
    log_and_print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))  # 打印并记录平均训练准确率
    log_and_print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))  # 打印并记录测试准确率

    runtime = time.time() - start_time  # 计算总运行时间
    log_and_print('\n Total Run Time: {0:0.4f}'.format(runtime))  # 打印并记录总运行时间

    # 保存训练日志到CSV文件
    data_log = {'Train Loss': train_loss, 'Train Accuracy': train_accuracy,
                'Test Loss': test_loss, 'Test Accuracy': test_acc}
    record = pd.DataFrame(data_log)
    record.to_csv('../log/MNIST/fed_{}_{}_E[{}]_iid[{}{}]_CR[{}]_epsilon[{}].csv'.format(args.dataset, args.model, args.epochs, args.iid, args.unequal, args.compression_ratio, args.epsilon))

    # 绘制平均准确率曲线并保存图像
    matplotlib.use('Agg')
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(list(range(len(train_accuracy))), train_accuracy)
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('../save/MNIST/fed_{}_{}_E[{}]_iid[{}{}]_CR[{}]_epsilon[{}].png'.
                format(args.dataset, args.model, args.epochs, args.iid, args.unequal, args.compression_ratio, args.epsilon))

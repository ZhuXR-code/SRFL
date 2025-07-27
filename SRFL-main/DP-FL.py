"""
这个是基于差分隐私的联邦学习main文件，跑这个文件需要修改options.py里面的--epsilon参数，这个参数是噪声参数
"""
import logging  # 日志模块，用于记录程序运行过程中的信息
import os  # 操作系统接口模块，用于获取文件名等操作
import time  # 时间模块，用于记录程序运行时间
from tqdm import tqdm  # 进度条库，用于显示训练进度
import copy  # 深拷贝模块，用于复制对象
import math  # 数学模块，提供数学函数

import torch  # PyTorch深度学习框架
torch.set_default_tensor_type(torch.DoubleTensor)  # 设置默认张量类型为双精度浮点型
import numpy as np  # NumPy库，用于科学计算
from scipy.fft import dct, idct  # 离散余弦变换及其逆变换，用于压缩和解压缩权重
import matplotlib  # Matplotlib绘图库
import matplotlib.pyplot as plt  # Matplotlib绘图接口
import pandas as pd  # Pandas库，用于数据处理和保存日志

# 导入自定义模块
from utils.options import args_parser, exp_details  # 参数解析和实验详情打印
from utils.dataset import get_dataset  # 数据集加载
from utils.models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar  # 模型定义
from utils.localupdate import LocalUpdate  # 本地更新模块
from utils.globalupdate import average_weights, test_inference  # 全局权重聚合和测试推理

# 获取当前文件名
current_file_name = os.path.basename(__file__)  # 使用os模块获取当前脚本的文件名
# 配置日志记录器
logging.basicConfig(
    filename=f'{current_file_name}.log',  # 日志文件名与当前文件名相同
    level=logging.INFO,  # 设置日志级别为INFO
    format='%(asctime)s - %(levelname)s - %(message)s'  # 日志格式：时间 - 日志级别 - 消息
)

# 定义通信网速（单位：MB/s）
COMMUNICATION_SPEED_MBPS = 5  # 假设通信网速为10 MB/s

def calculate_model_size(model):
    """计算模型大小（MB）"""
    total_params = sum(p.numel() for p in model.parameters())
    # total_size = total_params * model.parameters().__iter__().__next__().element_size() / (1024 * 1024)  # 返回模型大小（MB）
    total_size = total_params * model.parameters().__iter__().__next__().element_size() / 1024  # 返回模型大小（KB）
    return total_size


if __name__ == '__main__':
    start_time = time.time()  # 记录程序开始时间
    logging.info(f'======start_time======: {time.strftime("%Y-%m-%d %H:%M:%S")}')  # 打印并记录开始时间

    # 解析参数并打印实验详情
    args = args_parser()  # 解析命令行参数
    exp_details(args)  # 打印实验详情
    logging.info(f'Experiment Details: {args}')  # 记录实验参数

    # 记录设备信息
    device_start_time = time.time()  # 记录设备检测开始时间
    device = 'cuda' if args.gpu else 'cpu'  # 根据参数选择使用GPU还是CPU
    logging.info(f'Device: {device}')  # 记录设备信息
    device_end_time = time.time()  # 记录设备检测结束时间
    logging.info(f'Device Detection Time: {device_end_time - device_start_time:.4f} seconds')  # 记录设备检测耗时

    # 数据加载时间统计
    data_load_start_time = time.time()  # 记录数据加载开始时间
    train_dataset, test_dataset, user_samples = get_dataset(args)  # 加载训练集、测试集和用户样本划分
    data_load_end_time = time.time()  # 记录数据加载结束时间
    data_load_time = data_load_end_time - data_load_start_time  # 计算数据加载耗时
    logging.info(f'Datasets loaded: Train={len(train_dataset)}, Test={len(test_dataset)}, Users={len(user_samples)}')  # 打印数据集信息
    logging.info(f'Data Loading Time: {data_load_time:.4f} seconds')  # 记录数据加载耗时

    # 全局模型初始化时间统计
    model_init_start_time = time.time()  # 记录模型初始化开始时间
    global_model = None  # 初始化全局模型变量
    if args.model == 'cnn':  # 如果模型类型为CNN
        if args.dataset == 'mnist':  # 如果数据集为MNIST
            global_model = CNNMnist(args=args)  # 初始化MNIST的CNN模型
        elif args.dataset == 'fmnist':  # 如果数据集为Fashion-MNIST
            global_model = CNNFashion_Mnist(args=args)  # 初始化Fashion-MNIST的CNN模型
        elif args.dataset == 'cifar':  # 如果数据集为CIFAR-10
            global_model = CNNCifar(args=args)  # 初始化CIFAR-10的CNN模型
    elif args.model == 'mlp':  # 如果模型类型为MLP
        img_size = train_dataset[0][0].shape  # 获取输入图像的尺寸
        len_in = 1  # 初始化输入维度
        for x in img_size:
            len_in *= x  # 计算输入维度大小
        global_model = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes)  # 初始化MLP模型
    else:
        logging.error('Error: unrecognized model')  # 如果模型类型不被识别，记录错误日志
        exit('Error: unrecognized model')  # 退出程序

    global_model.to(device)  # 将模型移动到指定设备（CPU或GPU）
    global_model.train()  # 设置模型为训练模式
    model_init_end_time = time.time()  # 记录模型初始化结束时间
    model_init_time = model_init_end_time - model_init_start_time  # 计算模型初始化耗时
    logging.info(f'Global Model initialized: {args.model}')  # 记录模型初始化完成信息
    logging.info(f'Model Initialization Time: {model_init_time:.4f} seconds')  # 记录模型初始化耗时

    # 初始化全局权重
    global_weights = global_model.state_dict()  # 获取全局模型的权重字典
    global_weights_ = copy.deepcopy(global_weights)  # 深拷贝全局权重以备后续使用

    # 初始化训练损失、准确率等变量
    train_loss, train_accuracy = [], []  # 记录训练损失和准确率
    upload_times, download_times = [], []  # 用于记录上传和下载时间

    weights_numbers = copy.deepcopy(global_weights)  # 深拷贝权重用于后续计算

    # 开始训练循环，epoch表示通信轮数
    for epoch in tqdm(list(range(args.epochs))):  # tqdm显示进度条
        epoch_start_time = time.time()  # 记录当前epoch开始时间

        local_weights, local_losses = [], []  # 用于存储客户端的本地权重和损失
        local_weights_ = []  # 临时存储本地权重（未使用）

        print(f'\n | Global Training Round : {epoch + 1} |\n')  # 打印当前通信轮次
        logging.info(f'\n | Global Training Round : {epoch + 1} |\n')  # 记录当前通信轮次

        global_model.train()  # 设置模型为训练模式
        m = max(int(args.frac * args.num_users), 1)  # 计算参与训练的客户端数量
        index_of_users = np.random.choice(list(range(args.num_users)), m, replace=False)  # 随机选择客户端
        index = 0  # 客户端索引

        client_training_start_time = time.time()  # 记录客户端训练开始时间
        for k in index_of_users:  # 遍历选中的客户端
            local_model = LocalUpdate(args=args, dataset=train_dataset, index_of_samples=user_samples[k])  # 初始化本地更新模块

            w, loss = local_model.local_train(model=copy.deepcopy(global_model), global_round=epoch)  # 客户端本地训练

            local_weights_.append(copy.deepcopy(w))  # 存储本地权重（未使用）

            if args.model == 'cnn':  # 如果模型类型为CNN
                dp_noise_start_time = time.time()  # 记录差分隐私噪声添加开始时间
                for key, _ in list(w.items()):  # 遍历权重字典
                    N = w[key].numel()  # 获取权重张量的元素总数
                    weights_numbers[key] = torch.tensor(N)  # 记录权重元素总数
                    M = max(int(args.compression_ratio * N), 1)  # 计算压缩后的元素数量

                    w_dct = dct(w[key].numpy().reshape((-1, 1)))  # 对权重进行离散余弦变换
                    e = epoch  # 当前通信轮次
                    if e >= int(N / M):  # 如果轮次超过最大分割次数
                        e = e - int(N / M) * int(epoch / int(N / M))  # 调整轮次
                    y = w_dct[e * M:min((e + 1) * M, N), :]  # 提取当前轮次的权重片段

                    epsilon_user = args.epsilon + np.zeros_like(y)  # 设置差分隐私噪声参数

                    min_weight = min(y)  # 权重片段的最小值
                    max_weight = max(y)  # 权重片段的最大值
                    center = (max_weight + min_weight) / 2  # 权重片段的中心值
                    radius = (max_weight - center) if (max_weight - center) != 0. else 1  # 权重片段的半径
                    miu = y - center  # 权重片段相对于中心的偏移
                    Pr = (np.exp(epsilon_user) - 1) / (2 * np.exp(epsilon_user))  # 计算概率
                    u = np.zeros_like(y)  # 初始化随机变量
                    for i in range(len(y)):  # 遍历权重片段
                        u[i, 0] = np.random.binomial(1, Pr[i, :])  # 生成随机变量

                    for i in range(len(y)):  # 根据随机变量调整权重片段
                        if u[i, 0] > 0:
                            y[i, :] = center + miu[i, :] * ((np.exp(epsilon_user[i, :]) + 1) / (np.exp(epsilon_user[i, :]) - 1))
                        else:
                            y[i, :] = center + miu[i, :] * ((np.exp(epsilon_user[i, :]) - 1) / (np.exp(epsilon_user[i, :]) + 1))

                    w[key] = torch.from_numpy(y)  # 将调整后的权重片段转换为张量
                dp_noise_end_time = time.time()  # 记录差分隐私噪声添加结束时间
                dp_noise_time = dp_noise_end_time - dp_noise_start_time  # 计算差分隐私噪声添加耗时
                logging.info(f'DP Noise Addition Time for Client {k}: {dp_noise_time:.4f} seconds')  # 记录差分隐私噪声添加耗时

            local_weights.append(copy.deepcopy(w))  # 存储本地权重
            local_losses.append(copy.deepcopy(loss))  # 存储本地损失

            print(f'| client-{index + 1} : {k} finished!!! |')  # 打印客户端完成信息
            logging.info(f'| client-{index + 1} : {k} finished!!! |')  # 记录客户端完成信息
            index += 1  # 更新客户端索引

        client_training_end_time = time.time()  # 记录客户端训练结束时间
        client_training_time = client_training_end_time - client_training_start_time  # 计算客户端训练总耗时
        print(f'\n | Client Training End!!! | \n')  # 打印客户端训练结束信息
        logging.info(f'\n | Client Training End!!! | \n')  # 记录客户端训练结束信息
        logging.info(f'Client Training Time for Epoch {epoch + 1}: {client_training_time:.4f} seconds')  # 记录客户端训练耗时

        shapes_global = copy.deepcopy(global_weights)  # 深拷贝全局权重形状
        for key, _ in list(shapes_global.items()):  # 遍历权重字典
            shapes_global[key] = shapes_global[key].shape  # 记录权重形状

        aggregation_start_time = time.time()  # 记录权重聚合开始时间
        partial_global_weights = average_weights(local_weights)  # 聚合本地权重
        aggregation_end_time = time.time()  # 记录权重聚合结束时间
        aggregation_time = aggregation_end_time - aggregation_start_time  # 计算权重聚合耗时
        logging.info(f'Weight Aggregation Time for Epoch {epoch + 1}: {aggregation_time:.4f} seconds')  # 记录权重聚合耗时

        avg_loss = sum(local_losses) / len(local_losses)  # 计算平均训练损失
        train_loss.append(avg_loss)  # 记录平均训练损失
        print(f'Avg Training Loss for Epoch {epoch + 1}: {avg_loss}')  # 打印平均训练损失
        logging.info(f'Avg Training Loss for Epoch {epoch + 1}: {avg_loss}')  # 记录平均训练损失

        global_aggregation_start_time = time.time()  # 记录全局聚合开始时间
        for key, _ in partial_global_weights.items():  # 遍历聚合后的权重
            N = weights_numbers[key].item()  # 获取权重元素总数
            M = max(int(args.compression_ratio * N), 1)  # 计算压缩后的元素数量
            rec_matrix = np.zeros((N, 1))  # 初始化重构矩阵
            e = epoch  # 当前通信轮次
            if e >= int(N / M):  # 如果轮次超过最大分割次数
                e = e - int(N / M) * int(epoch / int(N / M))  # 调整轮次
            rec_matrix[e * M:min((e + 1) * M, N), :] = partial_global_weights[key]  # 填充重构矩阵
            x_rec = idct(rec_matrix)  # 对重构矩阵进行逆离散余弦变换
            global_weights_1D = global_weights[key].numpy().reshape((-1, 1))  # 将全局权重展平为一维
            global_weights_1D[e * M:min((e + 1) * M, N), :] = (global_weights_1D[e * M:min((e + 1) * M, N), :] + x_rec[e * M:min((e + 1) * M, N), :]) / 2  # 更新全局权重
            global_weights[key] = torch.from_numpy(global_weights_1D.reshape(shapes_global[key]))  # 将更新后的权重恢复形状

            print('key: ', key, '\t global_weights: ', global_weights[key].shape)  # 打印全局权重形状
            logging.info(f'key: {key} \t global_weights: {global_weights[key].shape}')  # 记录全局权重形状
        global_aggregation_end_time = time.time()  # 记录全局聚合结束时间
        global_aggregation_time = global_aggregation_end_time - global_aggregation_start_time  # 计算全局聚合耗时
        logging.info(f'Global Aggregation Time for Epoch {epoch + 1}: {global_aggregation_time:.4f} seconds')  # 记录全局聚合耗时

        global_model.load_state_dict(global_weights)  # 加载更新后的全局权重
        print(f'\n | Global Training Round : {epoch + 1} finished!!!!!!!!|\n')  # 打印全局训练轮次完成信息
        logging.info(f'\n | Global Training Round : {epoch + 1} finished!!!!!!!!|\n')  # 记录全局训练轮次完成信息

        evaluation_start_time = time.time()  # 记录评估开始时间
        list_acc, list_loss = [], []  # 初始化评估准确率和损失列表
        global_model.eval()  # 设置模型为评估模式
        for k in range(args.num_users):  # 遍历所有客户端
            local_model = LocalUpdate(args=args, dataset=train_dataset, index_of_samples=user_samples[k])  # 初始化本地更新模块
            acc, loss = local_model.inference(model=global_model)  # 在本地数据上进行推理
            list_acc.append(acc)  # 记录准确率
            list_loss.append(loss)  # 记录损失
        train_accuracy.append(sum(list_acc) / len(list_acc))  # 计算平均准确率
        evaluation_end_time = time.time()  # 记录评估结束时间
        evaluation_time = evaluation_end_time - evaluation_start_time  # 计算评估耗时
        logging.info(f'Evaluation Time for Epoch {epoch + 1}: {evaluation_time:.4f} seconds')  # 记录评估耗时

        print(f'\nAvg Training States after {epoch + 1} global rounds:')  # 打印平均训练状态
        print(f'Avg Training Loss : {train_loss[-1]}')  # 打印平均训练损失
        print('Avg Training Accuracy : {:.2f}% \n'.format(100 * train_accuracy[-1]))  # 打印平均训练准确率
        logging.info(f'\nAvg Training States after {epoch + 1} global rounds:')  # 记录平均训练状态
        logging.info(f'Avg Training Loss : {train_loss[-1]}')  # 记录平均训练损失
        logging.info('Avg Training Accuracy : {:.2f}% \n'.format(100 * train_accuracy[-1]))  # 记录平均训练准确率

        # 计算通信时间
        model_size = calculate_model_size(global_model)  # 计算模型大小
        logging.info(f'model_size for Epoch {epoch + 1}: {model_size}KB')
        upload_time = model_size / COMMUNICATION_SPEED_MBPS  # 上传时间
        download_time = model_size / COMMUNICATION_SPEED_MBPS  # 下载时间
        upload_times.append(upload_time)
        download_times.append(download_time)
        logging.info(f'Upload Time for Epoch {epoch + 1}: {upload_time:.4f} seconds')
        logging.info(f'Download Time for Epoch {epoch + 1}: {download_time:.4f} seconds')

        if math.isnan(train_loss[-1]):  # 如果训练损失为NaN
            train_loss.pop()  # 移除最后一个训练损失
            train_accuracy.pop()  # 移除最后一个训练准确率
            break  # 结束训练循环

        epoch_end_time = time.time()  # 记录当前epoch结束时间
        epoch_time = epoch_end_time - epoch_start_time  # 计算当前epoch总耗时
        logging.info(f'Epoch {epoch + 1} Total Time: {epoch_time:.4f} seconds')  # 记录当前epoch总耗时

    test_inference_start_time = time.time()  # 记录测试推理开始时间
    test_acc, test_loss = test_inference(args, global_model, test_dataset)  # 在测试集上进行推理
    test_inference_end_time = time.time()  # 记录测试推理结束时间
    test_inference_time = test_inference_end_time - test_inference_start_time  # 计算测试推理耗时
    logging.info(f'Test Inference Time: {test_inference_time:.4f} seconds')  # 记录测试推理耗时

    print(f' \n Results after {args.epochs} global rounds of training:')  # 打印最终训练结果
    print(f'|---- Avg Train Loss: {np.mean(np.array(train_loss))}')  # 打印平均训练损失
    print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))  # 打印平均训练准确率
    print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))  # 打印测试准确率
    logging.info(f' \n Results after {args.epochs} global rounds of training:')  # 记录最终训练结果
    logging.info(f'|---- Avg Train Loss: {np.mean(np.array(train_loss))}')  # 记录平均训练损失
    logging.info("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))  # 记录平均训练准确率
    logging.info("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))  # 记录测试准确率

    runtime = time.time() - start_time  # 计算程序总运行时间
    print(('\n Total Run Time: {0:0.4f}'.format(runtime)))  # 打印程序总运行时间
    logging.info(('\n Total Run Time: {0:0.4f}'.format(runtime)))  # 记录程序总运行时间

    # 将训练日志保存为CSV文件
    data_log = {
        'Train Loss': train_loss,
        'Train Accuracy': train_accuracy,
        'Test Loss': test_loss,
        'Test Accuracy': test_acc,
        'Upload Time': upload_times,
        'Download Time': download_times
    }
    record = pd.DataFrame(data_log)
    record.to_csv('../log/MNIST/fed_DP_{}_{}_E[{}]_iid[{}{}]_CR[{}]_epsilon[{}].csv'.
                  format(args.dataset, args.model, args.epochs, args.iid, args.unequal, args.compression_ratio, args.epsilon))

    matplotlib.use('Agg')  # 设置Matplotlib后端为非交互模式
    # 绘制并保存平均准确率随通信轮数的变化图
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')  # 图表标题
    plt.plot(list(range(len(train_accuracy))), train_accuracy)  # 绘制曲线
    plt.ylabel('Average Accuracy')  # y轴标签
    plt.xlabel('Communication Rounds')  # x轴标签
    plt.savefig('../save/MNIST/fed_DP_{}_{}_E[{}]_iid[{}{}]_CR[{}]_epsilon[{}].png'.
                format(args.dataset, args.model, args.epochs, args.iid, args.unequal, args.compression_ratio, args.epsilon))

import copy
import torch
from torch import nn
from torch.utils.data import DataLoader


def average_weights(local_weights):
    """聚合多个本地模型的权重参数，执行加权平均操作

    参数:
        local_weights (list[collections.OrderedDict]):
            包含多个有序字典的列表，每个字典对应一个本地模型的权重参数。
            字典的键为层名称，值为对应的PyTorch张量权重。

    返回值:
        collections.OrderedDict: 包含所有输入权重平均结果的有序字典，
        键值对结构与输入字典保持完全一致

    实现说明:
        该方法通常用于联邦学习中的模型聚合阶段，通过计算各客户端模型的参数平均值
        来更新全局模型参数
    """
    # 深拷贝第一个模型的权重作为基准，避免修改原始数据
    avg_weights = copy.deepcopy(local_weights[0])

    # 遍历神经网络的所有参数层
    for key in avg_weights.keys():
        # 累加所有本地模型在当前层的参数值
        for i in range(1, len(local_weights)):
            avg_weights[key] += local_weights[i][key]

        # 计算当前层参数的算术平均值（各客户端权重相等情况）
        avg_weights[key] = torch.div(avg_weights[key], len(local_weights))

    return avg_weights

def weighted_average_weights(w_list, sample_sizes):
    total_samples = sum(sample_sizes)
    return {
        k: sum(w[k] * (s/total_samples) for w, s in zip(w_list, sample_sizes))
        for k in w_list[0].keys()
    }


def test_inference(args, model, test_dataset):
    """
    在测试集上评估模型性能

    Args:
        args: 配置参数对象，需包含gpu属性(bool类型)
        model: 训练好的神经网络模型
        test_dataset: 测试数据集

    Returns:
        tuple: (准确率, 总损失值) 的元组，均为浮点数格式
    """
    # 切换到模型评估模式
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    # 设置计算设备（GPU/CPU）和损失函数
    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)

    # 创建测试数据加载器（固定batch顺序）
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # 遍历所有测试数据批次
    for batch_idx, (images, labels) in enumerate(test_loader):
        # 将数据移至指定设备
        images, labels = images.to(device), labels.to(device)

        # 前向传播计算输出
        outputs = model(images)
        # 计算并累加当前批次损失
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # 统计正确预测数量
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        # 累计样本总数
        total += len(labels)

    # 计算整体准确率
    accuracy = correct / total
    return accuracy, loss


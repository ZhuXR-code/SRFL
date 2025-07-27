import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    # 联邦学习相关的参数（参数符号遵循论文中的符号）
    # 设置训练的轮数
    parser.add_argument('--epochs', type=int, default=200,
                        help="number of rounds of training")
    
    
    # parser.add_argument('--num_users', type=int, default=2,
    #                     help="number of users: K")
    # 设置用户数量
    parser.add_argument('--num_users', type=int, default=50,
                        help="number of users: K")
    
    
    
    # 设置客户端的参与比例
    parser.add_argument('--frac', type=float, default=1,
                        help='the fraction of clients: C')
    # 设置本地训练的轮数
    # parser.add_argument('--local_ep', type=int, default=10,
    #                     help="the number of local epochs: E")
    parser.add_argument('--local_ep', type=int, default=10,
                        help="the number of local epochs: E")
    # 设置本地训练的批量大小
    parser.add_argument('--local_bs', type=int, default=64,
                        help="local batch size: B")
    # 设置学习率
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    # 设置SGD的动量
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')


    # model arguments
    # 模型相关的参数
    # 指定模型名称
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    # parser.add_argument('--model', type=str, default='mlp', help='model name')


    # 设置每种内核的数量
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    # 设置卷积使用的内核大小，以逗号分隔
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution')
    # 设置图像的通道数
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")
    # 设置归一化方法
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    # 设置卷积网络的过滤器数量
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    # 设置是否使用最大池化而不是步长卷积
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")

    # other arguments
    # 其他参数
    # 指定数据集名称
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset")
    # parser.add_argument('--dataset', type=str, default='fmnist', help="name \
    #                         of dataset")
    # parser.add_argument('--dataset', type=str, default='cifar', help="name \
    #                             of dataset")



    # 设置类别数量
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    # 设置GPU ID，如果要使用CUDA，否则默认使用CPU
    parser.add_argument('--gpu', default=None, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    # 指定优化器类型
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    # 设置IID或非IID数据划分方式
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    # 设置非IID情况下是否使用不均匀的数据划分
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    # 设置早停的轮数
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    # 设置是否详细输出
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    # 设置随机种子
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    # DP arguments   no effect
    # 差分隐私（DP）相关的参数，当前无效果
    # 安全参数，用于控制噪声的引入程度
    # parser.add_argument('--epsilon', type=float, default=1, help='privacy budget in DP')
    parser.add_argument('--epsilon', type=float, default=2, help='privacy budget in DP')    # 动态差分隐私，调整引入的噪声的参数大小，数字越大引入噪声越大，越小噪声越小但是安全性越低。试一下3-5

    # CS arguments  no effect
    # 压缩感知（CS）相关的参数，当前无效果
    # 设置压缩比
    # parser.add_argument('--compression_ratio', type=float, default=1)  # 压缩比比值是1，说明没有压缩模型。压缩比对模型精度影响大。压缩感知
    parser.add_argument('--compression_ratio', type=float, default=0.05) 

    # 解析命令行参数
    args = parser.parse_args()
    return args

def exp_details(args):
    print('Experimental details:')
    print(f'    Model           : {args.model}')
    print(f'    Optimizer       : {args.optimizer}')
    print(f'    Learning-rate   : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}')
    print(f'    Num Users       : {args.num_users}')
    print(f'    Dataset         : {args.dataset}')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}')
    return

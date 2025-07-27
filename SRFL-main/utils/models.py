from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self,dim_in,dim_hidden,dim_out):
        super(MLP,self).__init__()
        self.layer_hidden = nn.Linear(dim_in,dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_out = nn.Linear(dim_hidden,dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,X):
        X = X.view(-1,X.shape[1] * X.shape[-2] * X.shape[-1])
        X = self.layer_hidden(X)
        X = self.dropout(X)
        X = self.relu(X)
        X = self.layer_out(X)
        return self.softmax(X)

class CNNMnist(nn.Module):
    """
    定义了一个用于MNIST手写数字识别的卷积神经网络模型。

    参数:
        args (argparse.Namespace): 包含模型参数的对象，例如输入通道数和类别数。

    属性:
        conv1 (nn.Conv2d): 第一个卷积层，接受单通道输入图像并应用10个5x5大小的滤波器。
        conv2 (nn.Conv2d): 第二个卷积层，接受第一个卷积层的输出并应用20个5x5大小的滤波器。
        conv2_drop (nn.Dropout2d): 应用于第二个卷积层输出的dropout层，用于防止过拟合。
        fc1 (nn.Linear): 第一个全连接层，将展平的320维输入映射到50维输出。
        fc2 (nn.Linear): 第二个全连接层，将50维输入映射到10维输出，对应于MNIST的10个类别。
    """
    ''' def __init__(self,args):
        super(CNNMnist, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(args.num_channels, 32, kernel_size=(5, 5)),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(5, 5)),
            nn.MaxPool2d(2))
        self.full_connection = nn.Linear(64 * 4 * 4, args.num_classes)
    '''
    def __init__(self,args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

   
    #def __init__(self,args):
    #    super(CNNMnist, self).__init__()
    #    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    #    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    #    self.conv2_drop = nn.Dropout2d()
    #    self.fc1 = nn.Linear(320, 50)
    #    self.fc2 = nn.Linear(50, 10)

    '''def forward(self, X):
        X = self.layer1(X)
        X = self.layer2(X)
        X = X.view(-1, X.shape[1] * X.shape[2] * X.shape[3])
        X = self.full_connection(X)
        X = F.relu(X)
        return F.softmax(X)'''

    def forward(self, x):
        """
        前向传播函数，定义了输入数据x通过网络的前向传播过程

        参数:
        x (Tensor): 输入的数据张量

        返回:
        Tensor: 经过网络前向传播后的输出张量
        """
        # 第一个卷积层后使用最大池化和ReLU激活函数
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # 第二个卷积层加入Dropout以防止过拟合，随后使用最大池化和ReLU激活函数
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # 将卷积层输出展平为全连接层的输入
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        # 全连接层后使用ReLU激活函数
        x = F.relu(self.fc1(x))
        # 使用Dropout以防止过拟合，根据training状态决定是否执行
        x = F.dropout(x, training=self.training)
        # 最后一层全连接层输出
        x = self.fc2(x)
        # 使用Log Softmax作为输出层，适用于多分类问题
        return F.log_softmax(x, dim=1)

    #def forward(self, x):
     #   x = F.relu(F.max_pool2d(self.conv1(x), 2))
      #  x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
      #  x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
       # x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        #x = self.fc2(x)
        #return F.log_softmax(x, dim=1)


class CNNFashion_Mnist(nn.Module):
    """
    定义一个用于Fashion-Mnist数据集的卷积神经网络类。

    该网络包含两个卷积层，每个卷积层后接最大池化层，以及一个全连接层。

    参数:
    - args: 包含模型配置的参数，如输入通道数(num_channels)和输出类别数(num_classes)。
    """

    def __init__(self, args):
        """
        初始化CNNFashion_Mnist类。

        构建网络结构，包括两层卷积层和一层全连接层。
        """
        super(CNNFashion_Mnist, self).__init__()
        # 定义第一层卷积层和最大池化层
        self.layer1 = nn.Sequential(
            nn.Conv2d(args.num_channels, 32, kernel_size=(5, 5)),
            nn.MaxPool2d(2))
        # 定义第二层卷积层和最大池化层
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(5, 5)),
            nn.MaxPool2d(2))
        # 定义全连接层，输入为前一层的输出特征图展平后的大小，输出为类别数
        self.full_connection = nn.Linear(64 * 4 * 4, args.num_classes)

    def forward(self, X):
        """
        前向传播函数。

        参数:
        - X: 输入的特征，通常是图像数据。

        返回:
        - 经过网络处理后的softmax输出，表示输入属于各个类别的概率。
        """
        # 通过第一层卷积层和最大池化层
        X = self.layer1(X)
        # 通过第二层卷积层和最大池化层
        X = self.layer2(X)
        # 将特征图展平，以便输入到全连接层
        X = X.view(-1, X.shape[1] * X.shape[2] * X.shape[3])
        # 通过全连接层
        X = self.full_connection(X)
        # 应用ReLU激活函数
        X = F.relu(X)
        # 应用softmax函数，将输出转换为概率分布
        return F.softmax(X)


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar,self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(128 * 4 * 4, 1028)
        self.fc2 = nn.Linear(1028, args.num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


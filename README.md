# SRFL-FL 联邦学习项目

这是一个基于功能加密(Functional Encryption)的联邦学习实现，名为SRFL-FL。该项目结合了安全多方计算和联邦学习技术，通过加密方式保护客户端模型参数的隐私。

## 环境要求

### Python版本
- Python 3.6 
- Python 3.7
- Python 3.9 

### 核心依赖库
- PyTorch >= 1.8.0
- NumPy
- SciPy
- Matplotlib
- Pandas
- tqdm

### 硬件要求
- CPU \ GPU (GPU部分需要修改代码里面对应的指向参数)
- 内存 8GB+
- 存储空间 1GB+

## 项目搭建过程

### 1. 创建虚拟环境（可以创建，按需）
```bash
python -m venv srfl_fl_env
source srfl_fl_env/bin/activate  # Linux/Mac
# 或
srfl_fl_env\Scripts\activate  # Windows
```


### 2. 安装依赖
```bash
pip install torch numpy scipy matplotlib pandas tqdm
```


charm-crypto


### 3. 项目目录结构
```
srfl-FE-FL-clear/
├── FL-main/
│   ├── SRFL-main.py  # 主程序文件
│   └── utils/                       # 工具模块目录
│       ├── options.py               # 参数配置
│       ├── dataset.py               # 数据集处理
│       ├── models.py                # 模型定义
│       ├── localupdate.py           # 本地更新
│       └── globalupdate.py          # 全局更新
├── EDFE/                            # 功能加密模块
│   └── src/
│       └── inner_product/
│           └── single_input_fe/
│               └── elgamal_ip.py    # srfl实现
└── log/                             # 日志输出目录
└── save/                            # 结果保存目录
```


### 4. 数据集准备
项目支持以下数据集：
- MNIST
- CIFAR-10

数据集会自动下载到项目目录中。

## 运行命令

### 基本运行命令
```bash
python SRFL-main.py
```


### 主要参数说明
| 参数 | 默认值 | 说明 |
|------|--------|------|
| --dataset | mnist | 数据集类型 (mnist, fmnist, cifar) |
| --model | cnn | 模型类型 (cnn, mlp) |
| --epochs | 10 | 训练轮数 |
| --num_users | 10 | 参与训练的客户端数量 |
| --frac | 0.1 | 每轮参与训练的客户端比例 |
| --iid | 0 | 数据分布 (0: non-iid, 1: iid) |
| --lr | 0.01 | 学习率 |

## 功能特性

1. **功能加密保护**: 使用EDFE保护客户端模型参数
2. **多种模型支持**: 支持CNN和MLP模型
3. **多数据集支持**: 支持MNIST和CIFAR-10数据集
4. **性能监控**: 实时记录训练时间、通信开销和模型性能
5. **可视化输出**: 自动生成训练曲线和结果图表

## 输出文件

运行后将在以下位置生成输出文件：
- 日志文件: `SRFL-main.py.log`
- CSV数据: `../log/MNIST/` 目录下
- 图像文件: `../save/MNIST/` 目录下

## 注意事项

1. 首次运行时会自动下载所需数据集
2. 加密运算会增加计算开销，请确保有足够的计算资源
3. 项目默认使用CPU运行，如需使用GPU，请修改 `device = 'cpu'` 为相应的GPU配置
4. 输出目录需要预先创建或确保有写入权限

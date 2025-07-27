import math
from charm.toolbox.integergroup import IntegerGroupQ, integer
from typing import List, Dict, Tuple
from EDFE.src.helpers.additive_elgamal import AdditiveElGamal, ElGamalCipher
from EDFE.src.helpers.helpers import reduce_vector_mod, get_int
from EDFE.src.errors.wrong_vector_for_provided_key import WrongVectorForProvidedKey
import charm
import numpy as np
import hashlib
import random
import time

IntegerGroupElement = charm.core.math.integer.integer
ElGamalKey = Dict[str, IntegerGroupElement]

debug = True
p1 = integer(
    148829018183496626261556856344710600327516732500226144177322012998064772051982752493460332138204351040296264880017943408846937646702376203733370973197019636813306480144595809796154634625021213611577190781215296823124523899584781302512549499802030946698512327294159881907114777803654670044046376468983244647367)
q1 = integer(
    74414509091748313130778428172355300163758366250113072088661006499032386025991376246730166069102175520148132440008971704423468823351188101866685486598509818406653240072297904898077317312510606805788595390607648411562261949792390651256274749901015473349256163647079940953557388901827335022023188234491622323683)
p = 2 * p1 + 1
q = 2 * q1 + 1
N = p * q
N_squared = N ** 2

elgamal_group = IntegerGroupQ()
elgamal = AdditiveElGamal(elgamal_group, p, q)
elgamal_params = {"group": elgamal_group, "p": int(p), "q": int(q), "N": int(p) * int(q)}

pk, sk = elgamal.keygen(1024)
g1 = pk['g']
g = pow(g1, (2 * N), N_squared)

def output_p():
    return {"group": elgamal_group, "p": int(p)}

def H1(data: str, p: integer) -> integer:
    hash_object = hashlib.sha256(data.encode('utf-8'))
    r = int(hash_object.hexdigest(), 16)
    return r % int(p)

def set_up(security_parameter: int, vector_length: int) -> Tuple[List[ElGamalKey], List[ElGamalKey], int]:
    N = elgamal_params['N']
    master_public_key = [None] * vector_length
    master_secret_key = [None] * vector_length
    for i in range(vector_length):
        (master_public_key[i], master_secret_key[i]) = elgamal.keygen(secparam=security_parameter)
        master_public_key[i] = pow(g, master_secret_key[i]['x'], N_squared)
        master_public_key[i] = {'h': master_public_key[i]}
    return master_public_key, master_secret_key, int(N)

def KeyDerive(master_secret_key: List[ElGamalKey], master_public_key2: List[ElGamalKey], ctr: int, y: List[int], aux: str) -> integer:
    """
    根据主私钥、主公钥以及其他参数派生一个新的密钥。

    参数说明：
    - master_secret_key (List[ElGamalKey]): 主私钥列表，包含用于派生密钥的私钥信息。
    - master_public_key2 (List[ElGamalKey]): 第二组主公钥列表，用于生成中间密钥。
    - master_secret_key2 (List[ElGamalKey]): 第二组主私钥列表，辅助生成中间密钥。
    - ctr (int): 计数器，确保每次派生密钥的唯一性。
    - y (List[int]): 权重向量，用于加权计算最终派生密钥。
    - aux (str): 辅助字符串，增强派生密钥的安全性。

    返回值：
    - integer: 派生出的新的密钥值。
    """
    ctr += 1  # 更新计数器以确保唯一性
    skf = integer(0)  # 初始化派生密钥变量
    t1 = time.time()
    for i in range(len(y)):  # 遍历权重向量 y
        pk2i = master_public_key2[i]  # 获取第 i 个第二组主公钥
        # 使用主公钥和主私钥计算中间密钥
        key = pow(pk2i['h'], master_secret_key['x'], N_squared)
        # 使用哈希函数 H1 生成随机数 r
        r = H1(f"{key}{ctr}{aux}", elgamal_params['p'])
        # 累加加权后的随机数 r 到派生密钥中
        skf += r * y[i]
    print(f"KeyDerive {i} time = {time.time()-t1}")

    return skf  # 返回派生出的密钥


def Encrypt(master_public_key: dict, master_secret_key2: dict, ctr: int, X: List[int], aux: str, N: int) -> List[dict]:
    """
    加密函数，使用主公钥、主私钥和其他参数对输入向量进行加密。
    公式是  Ci = (1+Xi*N)*(Pk1^r)mod(N^2)

    参数说明：
    - master_public_key (dict): 主公钥，包含加密所需的公钥信息。
    - master_secret_key2 (dict): 第二个主私钥，用于生成加密过程中的中间密钥。
    - ctr (int): 计数器，用于确保每次加密的唯一性。
    - X (List[int]): 待加密的整数列表。
    - aux (str): 辅助字符串，用于增强加密的安全性。
    - N (int): 模数，用于模运算和加密计算。

    返回值：
    - List[dict]: 加密后的结果列表，每个元素是一个字典，包含两个键值对 "0" 和 "1"。
    """
    C = []  # 存储加密结果的列表
    ctr += 1  # 更新计数器以确保唯一性
    N_squared = integer(N ** 2)  # 计算 N 的平方，用于模运算
    pk1 = master_public_key['h']  # 提取主公钥中的公钥部分

    sk2i = master_secret_key2  # 使用第二个主私钥
    key = pow(pk1, sk2i['x'], N_squared)  # 计算中间密钥
    r = H1(f"{key}{ctr}{aux}", elgamal_params['p'])  # 使用哈希函数生成随机数 r
    r = int(r)  # 将 r 转换为整数类型
    # 计算加密的第二部分 term2
    term2 = pow(int(pk1), r, int(p))
    # 遍历输入向量 X，逐个加密每个元素
    for i in range(len(X)):
        # 计算加密的第一部分 term1
        term1 = (1 + X[i] * N) % N_squared
        # 构造加密结果 C_i 并添加到结果列表中
        C_i = {"0": int(term1), "1": int(term2)}
        C.append(C_i)

    return C  # 返回加密结果列表


def Decrypt(C: List[dict], y: List[int], N: integer) -> integer:
    # print(f'C:{C}')
    N_squared = integer(N ** 2)
    c32 = 1
    c42 = 1
    # len(C) 是ct的数量也就是用户端的数量
    for i in range(len(C)):
        # print(f'y[i]{y[i]}')
        c31 = C[i]['0'] ** y[i]
        c32 = (int(c32) * int(c31)) % N_squared
        c41 = C[i]['1'] ** y[i]
        c42 = int(c42 * (int(c41)) % p)
    E2 = int(c32) % N_squared
    result = (int(E2) - 1) // N
    return int(result)

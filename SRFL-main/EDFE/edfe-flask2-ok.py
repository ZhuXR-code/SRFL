from flask import Flask, request, jsonify, g
import src.inner_product.single_input_fe.elgamal_ip.edfe as edfe
import logging
import numpy as np

# 创建Flask应用实例
app = Flask(__name__)

# 配置日志记录
logging.basicConfig(level=logging.DEBUG)


# @app.before_request
# def before_request():
#     # 初始化 g 对象中的变量
#     g.pk = None
#     g.sk = None
#     g.N = None
#     g.num_keys = None
#
#
# @app.route('/setup', methods=['POST'])  # 修改为 POST 方法，支持传参
# def setup():
#     data = request.get_json()
#
#     # 获取用户输入的 num_keys，默认值为 6
#     num_keys = data.get('num_keys', 6)
#     if num_keys < 3:  # 确保 num_keys 至少为 3（至少需要 generator、Encryptor 和 decryptor）
#         return jsonify({'error': 'num_keys must be at least 3'}), 400
#
#     logging.debug("Setting up with key_size=1024, num_keys=%d", num_keys)
#     key_size = 1024
#     pk, sk, N = edfe.set_up(key_size, num_keys)
#
#     # 将生成的值存储在 g 对象中
#     g.pk = pk
#     g.sk = sk
#     g.N = N
#     g.num_keys = num_keys
#
#     return jsonify({
#         'pk': pk,
#         'sk': sk,
#         'num_keys': num_keys
#     })


@app.route('/process', methods=['POST'])
def process():
    '''
    输入：
    POST /process
    Content-Type: application/json

    {
        "num_keys": 8,
        "y": [1, 2, 3, 4, 5],
        "x": [1, 1, 1, 1, 1]
    }

    Returns:

    '''
    # 记录处理开始时的输入数据
    logging.debug("Processing with data: %s", request.get_json())

    # 获取请求中的JSON数据
    data = request.get_json()

    # 从数据中获取密钥数量，如果没有提供，则默认为8
    num_keys = data.get('num_keys', 8)
    # 定义密钥大小为1024位
    key_size = 1024
    # 调用EDFE的设置函数生成公钥和私钥
    pk, sk, N = edfe.set_up(key_size, num_keys)

    # 分配第一个密钥对为生成器
    pk1, sk1 = pk[0], sk[0]
    # 分配中间的密钥对为加密器
    pk2, sk2 = pk[1:num_keys - 2], sk[1:num_keys - 2]
    # 分配最后一个密钥对为解密器
    pk3, sk3 = pk[num_keys - 1], sk[num_keys - 1]

    # 从数据中获取聚合权重和加密者的明文
    y = data['y']
    x = data['x']

    # 初始化计数器为0
    ctr = 0
    # 定义辅助信息为'edfe'
    aux = 'edfe'

    # 使用密钥派生函数生成中间密钥
    skf = edfe.KeyDerive(sk1, pk2, sk2, ctr, y, aux)
    # 使用加密函数加密明文
    C_x = edfe.Encrypt(pk1, sk2, pk3, ctr, x, aux, N)
    # 使用解密函数解密密文
    result = edfe.Decrypt(pk1, skf, sk3, C_x, y, N)

    # 计算预期的内积结果，用于验证加密方案的正确性
    expected_inner_prod = np.inner(x, y)

    # 记录处理结束时的结果
    logging.debug("Processed result: %s", result)
    # 返回处理结果，包括EDFE结果和正确的结果
    return jsonify({
        'type':aux,
        'C_x':C_x,
        'EDFE result is': int(result),
        'The correct result is': int(expected_inner_prod)
    })


if __name__ == '__main__':
    app.run(debug=True)

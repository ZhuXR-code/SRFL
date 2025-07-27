from flask import Flask, request, jsonify
import src.inner_product.single_input_fe.elgamal_ip.edfe as edfe
import logging
import numpy as np
import xmltodict
from xml.etree.ElementTree import Element
import xmljson  # 导入第三方库
import json

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Element):
            return xmltodict.parse(obj)  # 使用 xmltodict 解析 Element
        elif isinstance(obj, (list, dict)):
            if isinstance(obj, dict):
                return {k: self.default(v) for k, v in obj.items()}
            else:
                return [self.default(i) for i in obj]
        elif isinstance(obj, str):
            # 处理包含 mod 操作的结果的字符串
            if 'mod' in obj:
                parts = obj.split(' mod ')
                value = int(parts[0].strip())
                modulus = int(parts[1].strip())
                return {'value': value, 'modulus': modulus}
            return obj
        elif hasattr(obj, '__dict__'):
            # 处理具有 __dict__ 属性的对象
            return self.default(obj.__dict__)
        return super().default(obj)



def convert_element(obj):
    if isinstance(obj, Element):
        return xmltodict.parse(obj)  # 使用 xmltodict 解析 Element
    elif isinstance(obj, list):
        return [convert_element(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: convert_element(v) for k, v in obj.items()}
    elif isinstance(obj, str):
        # 处理包含 mod 操作的结果的字符串
        if 'mod' in obj:
            parts = obj.split(' mod ')
            value = int(parts[0].strip())
            modulus = int(parts[1].strip())
            return {'value': value, 'modulus': modulus}
        return obj
    elif hasattr(obj, '__dict__'):
        # 处理具有 __dict__ 属性的对象
        return convert_element(obj.__dict__)
    return obj



# 创建Flask应用实例
app = Flask(__name__)
app.json_encoder = CustomJSONEncoder  # 注册编码器


# 配置日志记录
logging.basicConfig(level=logging.DEBUG)



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
    {
        "type": "edfe",
        "EDFE result is": <decrypted_result>,
        "The correct result is": <expected_inner_prod>
    }
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
        'type': aux,
        'EDFE result is': int(result),
        'The correct result is': int(expected_inner_prod)
    })


# @app.route('/encrypt', methods=['POST'])
# def encrypt():
#     '''
#     输入：
#     POST /encrypt
#     Content-Type: application/json
#
#     {
#         "num_keys": 8,
#         "y": [1, 2, 3, 4, 5],
#         "x": [1, 1, 1, 1, 1]
#     }
#
#     Returns:
#     {
#         "C_x": <encrypted_ciphertext>,
#         "pk1": <public_key_1>,
#         "skf": <derived_key>,
#         "sk3": <private_key_3>,
#         "y": <aggregation_weights>,
#         "N": <modulus>
#     }
#     '''
#     # 记录处理开始时的输入数据
#     # logging.debug("Encrypting with data: %s", request.get_json())
#
#     # 获取请求中的JSON数据
#     data = request.get_json()
#
#     # 从数据中获取密钥数量，如果没有提供，则默认为8
#     num_keys = data.get('num_keys', 8)
#     # 定义密钥大小为1024位
#     key_size = 1024
#     # 调用EDFE的设置函数生成公钥和私钥
#     pk, sk, N = edfe.set_up(key_size, num_keys)
#     print("原始 N 类型:", type(N))  # 确认 N 是否是 int
#     # logging.debug("Generated keys: pk=%s, sk=%s, N=%s", pk, sk, N)
#
#     # 分配第一个密钥对为生成器
#     pk1, sk1 = pk[0], sk[0]
#     # 分配中间的密钥对为加密器
#     pk2, sk2 = pk[1:num_keys - 2], sk[1:num_keys - 2]
#     # 分配最后一个密钥对为解密器
#     pk3, sk3 = pk[num_keys - 1], sk[num_keys - 1]
#     # logging.debug("Assigned keys: pk1=%s, sk1=%s, pk2=%s, sk2=%s, pk3=%s, sk3=%s", pk1, sk1, pk2, sk2, pk3, sk3)
#
#     # 从数据中获取加密者的明文
#     y = data['y']
#     x = data['x']
#     # logging.debug("Received data: y=%s, x=%s", y, x)
#
#     # 初始化计数器为0
#     ctr = 0
#     # 定义辅助信息为'edfe'
#     aux = 'edfe'
#
#     # 使用密钥派生函数生成中间密钥
#     skf = edfe.KeyDerive(sk1, pk2, sk2, ctr, y, aux)
#     # logging.debug("Derived key: skf=%s", skf)
#
#     # 使用加密函数加密明文
#     C_x = edfe.Encrypt(pk1, sk2, pk3, ctr, x, aux, N)
#     # logging.debug("Encrypted result: C_x=%s", C_x)
#
#     # skf = int(skf)
#
#     print(f"C_x 类型: {type(C_x)}")
#     print(f"pk1 类型: {type(pk1)}")
#     # print(f"skf 类型: {type(skf)}")
#     print(f"sk3 类型: {type(sk3)}")
#     print(f"y 类型: {type(y)}")
#     print(f"x 类型: {type(x)}")
#     print(f"N 类型: {type(N)}")
#
#     # # 转换 skf 为 JSON 可序列化的类型
#     # if isinstance(skf, Element):
#     #     skf = xmljson.badgerfish.data(skf)  # 转换为字典
#     # elif isinstance(skf, list) or isinstance(skf, dict):
#     #     skf = CustomJSONEncoder().default(skf)  # 递归处理嵌套结构
#
#     # 返回前打印 skf 类型
#     # print("转换后 skf 类型:", type(skf))
#
#     # 将必要的密钥和参数返回给客户端
#     return jsonify({
#         'C_x': C_x,
#         'pk1': pk1,
#         'skf': skf,
#         'sk3': sk3,
#         'y': y,
#         'x': x,
#         'N': N,
#     })

@app.route('/encrypt', methods=['POST'])
def encrypt():
    # 获取请求中的JSON数据
    data = request.get_json()

    # 从数据中获取密钥数量，如果没有提供，则默认为8
    num_keys = data.get('num_keys', 8)
    # 定义密钥大小为1024位
    key_size = 1024

    # 从数据中获取聚合权重和加密者的明文
    y = data['y']
    x = data['x']

    # 调用EDFE的设置函数生成公钥和私钥
    pk, sk, N = edfe.set_up(key_size, num_keys)
    # 分配第一个密钥对为生成器
    pk1, sk1 = pk[0], sk[0]
    # 分配中间的密钥对为加密器
    pk2, sk2 = pk[1:num_keys - 2], sk[1:num_keys - 2]
    # 分配最后一个密钥对为解密器
    pk3, sk3 = pk[num_keys - 1], sk[num_keys - 1]

    # 初始化计数器为0
    ctr = 0
    # 定义辅助信息为'edfe'
    aux = 'edfe'

    # # 使用密钥派生函数生成中间密钥
    # skf = edfe.KeyDerive(sk1, pk2, sk2, ctr, y, aux)
    # 使用加密函数加密明文
    C_x = edfe.Encrypt(pk1, sk2, pk3, ctr, x, aux, N)

    # skf = int(skf)
    # 打印最终类型验证
    # print(f"C_x 类型: {type(C_x)}")
    # print(f"pk1 类型: {type(pk1)}")
    # # print(f"skf 类型: {type(skf)}")
    # print(f"sk3 类型: {type(sk3)}")
    # print(f"y 类型: {type(y)}")
    # print(f"x 类型: {type(x)}")
    # print(f"N 类型: {type(N)}")

    return jsonify({
        'C_x': C_x,
        'y': y,
        'x': x,
        'N': N,
    })


@app.route('/decrypt', methods=['POST'])
def decrypt():
    '''
    输入：
    POST /decrypt
    Content-Type: application/json

    {
        "C_x": <encrypted_ciphertext>,
        "pk1": <public_key_1>,
        "skf": <derived_key>,
        "sk3": <private_key_3>,
        "y": <aggregation_weights>,
        "N": <modulus>
    }

    Returns:
    {
        "EDFE result is": <decrypted_result>,
        "The correct result is": <expected_inner_prod>
    }
    '''
    # 记录处理开始时的输入数据
    # logging.debug("Decrypting with data: %s", request.get_json())

    # 获取请求中的JSON数据
    data = request.get_json()

    # 从数据中获取密文和必要参数
    C_x = data['C_x']
    y = data['y']
    x = data['x']
    N = data['N']

    # 初始化计数器为0
    ctr = 0
    # 定义辅助信息为'edfe'
    aux = 'edfe'

    # 使用解密函数解密密文
    result = edfe.Decrypt(C_x, y, N)

    # 计算预期的内积结果，用于验证加密方案的正确性
    expected_inner_prod = np.inner(x, y)  # 根据需求调整

    # 记录处理结束时的结果
    # logging.debug("Decrypted result: %s", result)
    # 返回解密结果
    return jsonify({
        'EDFE result is': int(result),
        'The correct result is': int(expected_inner_prod)
    })


if __name__ == '__main__':
    app.run(debug=True)

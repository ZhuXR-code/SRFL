import requests
import json
import logging

# Flask 应用的 URL
BASE_URL = 'http://127.0.0.1:5000'

def test_process():
    # 定义处理请求的数据
    process_data = {
        "num_keys": 8,
        "y": [1, 2, 3, 4, 5],
        "x": [1, 1, 1, 1, 1]
    }

    # 发送 POST 请求到 /process 接口
    response = requests.post(f'{BASE_URL}/process', json=process_data)

    # 检查响应状态码
    if response.status_code == 200:
        result = response.json()
        edfe_result = result['EDFE result is']
        correct_result = result['The correct result is']
        logging.debug(f"Processed result: {edfe_result}, Correct result: {correct_result}")
        return edfe_result, correct_result
    else:
        logging.error(f"Process failed with status code {response.status_code}: {response.text}")
        return None, None

def test_encrypt():
    # 定义加密请求的数据
    encrypt_data = {
        "num_keys": 8,
        "y": [1, 2, 3, 4, 5],
        "x": [1, 1, 1, 1, 1]
    }

    # 发送 POST 请求到 /encrypt 接口
    response = requests.post(f'{BASE_URL}/encrypt', json=encrypt_data)

    # 检查响应状态码
    if response.status_code == 200:
        result = response.json()
        C_x = result['C_x']
        y = result['y']
        x = result['x']
        N = result['N']

        logging.debug(f"Encrypted result: {C_x}")
        return C_x, y, x, N
    else:
        logging.error(f"Encrypt failed with status code {response.status_code}: {response.text}")
        return None, None, None, None, None, None

def test_decrypt(C_x, y, x, N):
    N = int(N.get('text')) if isinstance(N, dict) else N  # 根据编码器返回的结构调整
    # 定义解密请求的数据
    decrypt_data = {
        "C_x": C_x,
        "y": y,
        "x": x,
        "N": N
    }

    # 发送 POST 请求到 /decrypt 接口
    response = requests.post(f'{BASE_URL}/decrypt', json=decrypt_data)

    # 检查响应状态码
    if response.status_code == 200:
        result = response.json()
        edfe_result = result['EDFE result is']
        correct_result = result['The correct result is']
        logging.debug(f"Decrypted result: {edfe_result}, Correct result: {correct_result}")
        return edfe_result, correct_result
    else:
        logging.error(f"Decrypt failed with status code {response.status_code}: {response.text}")
        return None, None

if __name__ == '__main__':
    # 配置日志记录
    logging.basicConfig(level=logging.DEBUG)

    # # 测试处理接口【测试通过】
    # edfe_result, correct_result = test_process()
    # if edfe_result is not None and correct_result is not None:
    #     assert edfe_result == correct_result, f"EDFE result {edfe_result} does not match correct result {correct_result}"
    #     logging.info("Process test passed.")
    # else:
    #     logging.error("Process test failed.")

    # 测试加密接口
    C_x, y, x, N = test_encrypt()
    if C_x and y and x and N:
        logging.info("Encrypt test passed.")
    else:
        logging.error("Encrypt test failed.")

    # 测试解密接口
    if C_x and y and x and N:
        edfe_result, correct_result = test_decrypt(C_x, y, x, N)
        if edfe_result is not None and correct_result is not None:
            assert edfe_result == correct_result, f"EDFE result {edfe_result} does not match correct result {correct_result}"
            logging.info("Decrypt test passed.")
        else:
            logging.error("Decrypt test failed.")
    else:
        logging.error("Cannot run decrypt test due to failed encrypt test.")

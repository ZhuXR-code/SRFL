from flask import Flask, request, jsonify
import src.inner_product.single_input_fe.elgamal_ip.edfe as edfe
import logging

# 创建Flask应用实例
app = Flask(__name__)

# 配置日志记录
logging.basicConfig(level=logging.DEBUG)

@app.route('/setup', methods=['GET'])
def setup():
    logging.debug("Setting up with key_size=%d, num_keys=%d", key_size, num_keys)
    key_size = request.args.get('key_size', default=1024, type=int)
    num_keys = request.args.get('num_keys', default=6, type=int)
    pk, sk, N = edfe.set_up(key_size, num_keys)
    logging.debug("Setup complete: pk=%s, sk=%s, N=%s", pk, sk, N)
    return jsonify({
        'pk': pk,
        'sk': sk,
        'N': N
    })

@app.route('/derive_key', methods=['POST'])
def derive_key():
    logging.debug("Deriving key with data: %s", request.get_json())
    data = request.get_json()
    sk1 = data['sk1']
    pk2 = data['pk2']
    sk2 = data['sk2']
    ctr = data['ctr']
    y = data['y']
    aux = data['aux']
    skf = edfe.KeyDerive(sk1, pk2, sk2, ctr, y, aux)
    logging.debug("Derived key: %s", skf)
    return jsonify({
        'skf': skf
    })

@app.route('/encrypt', methods=['POST'])
def encrypt():
    logging.debug("Encrypting with data: %s", request.get_json())
    data = request.get_json()
    pk1 = data['pk1']
    sk2 = data['sk2']
    pk3 = data['pk3']
    ctr = data['ctr']
    x = data['x']
    aux = data['aux']
    N = data['N']
    C_x = edfe.Encrypt(pk1, sk2, pk3, ctr, x, aux, N)
    logging.debug("Encrypted data: %s", C_x)
    return jsonify({
        'C_x': C_x
    })

@app.route('/decrypt', methods=['POST'])
def decrypt():
    logging.debug("Decrypting with data: %s", request.get_json())
    data = request.get_json()
    pk1 = data['pk1']
    skf = data['skf']
    sk3 = data['sk3']
    C_x = data['C_x']
    y = data['y']
    N = data['N']
    result = edfe.Decrypt(pk1, skf, sk3, C_x, y, N)
    logging.debug("Decrypted result: %s", result)
    return jsonify({
        'result': result
    })

if __name__ == '__main__':
    app.run(debug=True)

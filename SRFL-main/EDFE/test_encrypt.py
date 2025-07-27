import src.inner_product.single_input_fe.elgamal_ip.edfe as edfe
import numpy as np

# 设置密钥
key_size = 1024
num_keys = 6
pk, sk, N = edfe.set_up(key_size, num_keys)

# 分配公钥私钥
pk1, sk1 = pk[0], sk[0]  # generator
pk2, sk2 = [pk[1], pk[2], pk[3], pk[4], pk[5]], [sk[1], sk[2], sk[3], sk[4], sk[5]]  # Encryptor
pk3, sk3 = pk[4], sk[4]  # decryptor

# 聚合权重和明文
y = [1, 1, 1, 1, 1]  # 聚合权重
x = [1, 2, 3, 4, 4]  # 加密者的明文
ctr = 0  # 计数器
aux = 'edfe'  # 辅助信息

# 派生密钥
skf = edfe.KeyDerive(sk1, pk2, sk2, ctr, y, aux)

# 加密
C_x = edfe.Encrypt(pk1, sk2, pk3, ctr, x, aux, N)
print("Encrypted data:", C_x)

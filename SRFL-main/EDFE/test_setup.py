import src.inner_product.single_input_fe.elgamal_ip.edfe as edfe

key_size = 1024
num_keys = 6
pk, sk, N = edfe.set_up(key_size, num_keys)
print(pk, sk, N)
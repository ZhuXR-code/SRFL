from charm.toolbox.integergroup import IntegerGroup
import charm
from typing import List
from charm.core.math.integer import getMod, toInt
from random import SystemRandom
import numpy as np

from ..errors.vector_size_mismatch_error import VectorSizeMismatchError
from ..helpers.matrix import Matrix

IntegerGroupElement = charm.core.math.integer.integer
IntegerMatrix = List[List[int]]


def get_random_from_Zl(l: int) -> int:
    """
    Returns a random number from finite field of integers modulo l
    Args:
        l: modulus of ff

    Returns: random number from Zl

    """
    cryptogen = SystemRandom()
    return cryptogen.randrange(l)


def sample_random_matrix_mod_np(size: tuple, mod: int) -> np.ndarray:
    return np.random.randint(mod, size=size, dtype=np.longlong)


def sample_random_matrix_mod(size: tuple, mod: int) -> Matrix:
    matrix = Matrix(dims=size)
    for i in range(size[0]):
        for j in range(size[1]):
            matrix[i, j] = get_random_from_Zl(mod)
    return matrix


def multiply_matrices_mod_np(A: np.ndarray, B: np.ndarray, mod: int) -> np.ndarray:
    return np.mod(np.dot(A, B), mod)


def multiply_matrices_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.dot(a, b)


def sample_random_matrix_from_normal_dist(size: tuple, standard_dev):
    rng = np.random.default_rng()
    return np.round(rng.normal(loc=0, scale=standard_dev, size=size))


def add_vectors_mod(a: List[int], b: List[int], mod: int) -> List[int]:
    if len(a) != len(b):
        raise VectorSizeMismatchError
    n = len(a)
    out = [(a[i] + b[i]) % mod for i in range(n)]
    return out


# def generate_group(sec_param):
#     """Generates a Schnorr mod p where p is a prime of
#     bit-size equal to sec_param
#
#     Args:
#         sec_param (int): security parameter, bit-size of p
#
#     Returns:
#         Tuple(): _description_
#     """
#     group = IntegerGroup()
#     group.paramgen(sec_param)
#     g = group.randomGen()
#     return (group, g)


def generate_group(sec_param):
    """Generates a Schnorr mod p where p is a prime of
    bit-size equal to sec_param

    Args:
        sec_param (int): security parameter, bit-size of p

    Returns:
        Tuple(): _description_
    """
    group = IntegerGroup()
    group.paramgen(sec_param)
    return group


def get_random_generator(group):
    return group.randomGen()


def inner_product_group_vector(a: List[IntegerGroupElement], b: List[int]) -> int:
    """
    Calculates inner product of group element vector and integer vector
    Args:
        a: group elements vector
        b: integer vector

    Returns: inner product of the vectors

    """
    if len(a) != len(b):
        raise VectorSizeMismatchError
    n = len(a)
    inner = 0
    for i in range(n):
        inner += get_int(a[i]) * b[i]
    return inner


def inner_product_modulo(a: List[int], b: List[int], mod: int) -> int:
    if len(a) != len(b):
        raise VectorSizeMismatchError
    inner = 0
    for i in range(len(a)):
        inner += a[i] * b[i] % mod
    return inner


def inner_product(a: List[int], b: List[int]) -> int:
    if len(a) != len(b):
        raise VectorSizeMismatchError
    n = len(a)
    return sum([a[i] * b[i] for i in range(n)])


def inner_product_matrices(a: List[List[int]], b: List[List[int]]) -> int:
    if len(a) != len(b):
        raise VectorSizeMismatchError
    n = len(a)
    return sum([inner_product(a[i], b[i]) for i in range(n)])


def inner_product_matrices_mod(a: List[List[int]], b: List[List[int]], mod: int) -> int:
    if len(a) != len(b):
        raise VectorSizeMismatchError
    n = len(a)
    return sum([inner_product_modulo(a[i], b[i]) for i in range(n)])


def decode_vector_from_group_elements(vector: List[IntegerGroupElement], group: IntegerGroup) -> List[int]:
    decoded_vector = []
    for i in range(len(vector)):
        decoded_vector.append(bytes_to_int(group.decode(vector[i])))
    return decoded_vector


def decode_from_group_element(element: IntegerGroupElement, group: IntegerGroup) -> int:
    return bytes_to_int(group.decode(element))


def encode_vector_to_group_elements(vector: List[int], group: IntegerGroup) -> List[IntegerGroupElement]:
    encoded_vector = []
    for i in range(len(vector)):
        encoded_vector.append(encode_as_group_element(vector[i], group))
    return encoded_vector


def encode_as_group_element(x: int, group: IntegerGroup) -> IntegerGroupElement:
    return group.encode(int_to_bytes(x))


def int_to_bytes(x: int) -> bytes:
    return x.to_bytes((x.bit_length() + 7) // 8, byteorder='big')


def bytes_to_int(xbytes: bytes) -> int:
    return int.from_bytes(xbytes, byteorder='big')


def get_modulus(element: IntegerGroupElement) -> int:
    """From a mod N returns N

    Args:
        element (IntegerGroupElement): Group element of form a mod N

    Returns:
        int: Modulus of modular expression, a mod N -> N
    """
    mod = int(getMod(element))
    return mod


def get_int(element: IntegerGroupElement) -> int:
    """From a mod N returns a

    Args:
        element (IntegerGroupElement): Group element of form a mod N

    Returns:
        int: Integer part of modular expression, a mod N -> a
    """
    return int(toInt(element))


def product(vector: List[int]) -> int:
    prod = 1
    for element in vector:
        prod = prod * element
    return prod


def dummy_discrete_log(a: int, b: int, mod: int, limit: int) -> int:
    """Calculates discrete log of b in the base of a modulo mod, provided the
    result is smaller than limit. Otherwise, returns None

    Args:
        a (int): base of logarithm
        b (int): number from which the logarithm is calculated
        mod (int): modulus of logarithm 
        limit (int): limit within which the result should lie

    Returns:
        int: result of logarithm or None if the result was not found withn the limit
    """
    for i in range(limit):
        if pow(a, i, mod) == b:
            return i
    return None


def reduce_vector_mod(vector: List[int], mod: int) -> List[int]:
    """Reduces all elements of a vector modulo mod

    Args:
        vector (List[int]): list representation of vector
        mod (int): modulus

    Returns:
        List[int]: vector with reduced elements
    """
    reduced = []
    for i in range(len(vector)):
        reduced.append(vector[i] % mod)
    return reduced

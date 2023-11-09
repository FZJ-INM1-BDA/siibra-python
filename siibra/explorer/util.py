import math
import struct
from functools import wraps
from typing import Callable

cipher = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_-'
separator = '.'
neg = '~'
def encode_number(n, float_flag=False):
    if float_flag:
        b=struct.pack('f', n)
        new_n=struct.unpack('i',b)
        return encode_int(new_n[0])
    else:
        return encode_int(n)

def encode_int(n):
    if not isinstance(n, int):
        raise ValueError('Cannot encode int')

    residual=None
    result=''
    if n < 0:
        result += neg
        residual = n * -1
    else:
        residual = n
    
    while True:
        result = cipher[residual % 64] + result
        residual = math.floor(residual / 64)

        if residual == 0:
            break
    return result

def decode_int(n):
    neg_flag = False
    if n[-1] == neg:
        neg_flag = True
        n = n[:-1]

    result = 0
    for char in n:
        val = cipher.index(char)
        result = result * 64 + val

    if neg_flag:
        result = result * -1
    return result

def decode_number(n, float_flag=False):
    if float_flag:
        raise NotImplementedError
    return decode_int(n)


def post_process(post_process: Callable):
    def outer(fn):
        @wraps(fn)
        def inner(*args, **kwargs):
            val = fn(*args, **kwargs)
            return post_process(val, *args, **kwargs)
        return inner
    return outer

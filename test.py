import numpy as np

def AA(a, b):
    return a + b

def BB(aa, a, b):
    return aa(a, b)

print(BB(AA, 1, 2))
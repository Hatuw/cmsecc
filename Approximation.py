# -*- coding: utf-8 -*-
"""
The inference of rational approximation.
Topsec Cup 2018
Author: hatuw
Mail: jiaxi_wu@ieee.org
"""

import numpy as np

sequence_file = './sequence.txt'


# # load the sequence data
# with open(sequence_file, 'r') as sq_in:
#     # for test: only load the top 1000 numbers
#     sequence = sq_in.readline()[:1000]

test_seq = "0011100101"
sequence = test_seq

# find the first nonzero k(a_i)
for index, val in enumerate(sequence):
    if val == '1':
        k = index + 1
        break

# define phi(x)
phi = lambda x: np.max(np.abs(x))


# define minimize(list)
def find_d(*args):
    """
    d: d is among the odd integers immediately
    less than(<) or greater than(>)
    (f1-f2)/(g1-g2) and -(f1,f2)/(g1+g2)
    """
    lowwer_bound = args[2]+1 if args[2] % 2 == 0 else args[2]
    upper_bound = args[3]
    d = np.arange(lowwer_bound, upper_bound, 2)[:, np.newaxis]
    assert len(args) == 4, "It need tow args(f, g) to find d"
    d = phi(args[0] + d*args[1])
    return d


def minimize(f=None, d=None, g=None):
    # d % 2 ==0
    pass

# define alpha, f and g
a_k = 1 # $a_{k-1}$
alpha = a_k * (2**(k-1))
f = np.array([0, 2])
g = np.array([2**(k-1), 1])
print(a_k, alpha, f, g)

i = 0
while i < 10:
    # input a new bits a_k
    """
    for index, val in enumerate(sequence):
        if val == '1':
            k = index + 1
            break
    """

    alpha += a_k * (2**k)
    if alpha*g[1] - g[0] // 2**(k+1) == 0:
        f *= 2

    elif phi(g) < phi(f):
        # let d to be odd and minimize phi(f+dg)
        temp_bound1 = (f[0] - f[1]) / (g[0] - g[1])
        temp_bound2 = (f[0] + f[1]) / (g[0] + g[1])
        bound = np.sort([temp_bound1, temp_bound2])
        d = find_d(f, g, bound[0], bound[1])
        g = f + d*g
        f = 2 * g

    else:
        # let d to be odd and minimize phi(f+dg)
        temp_bound1 = (g[0] - g[1]) / (f[0] - f[1])
        temp_bound2 = (g[0] + g[1]) / (f[0] + f[1])
        bound = np.sort([temp_bound1, temp_bound2])
        d = find_d(g, f, bound[0], bound[1])
        g = f + d*f
        f *= 2
    k += 1
    i += 1
print(g)

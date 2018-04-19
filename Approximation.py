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
with open(sequence_file, 'r') as sq_in:
    # for test: only load the top 1000 numbers
    sequence = sq_in.readline()[:120]

# test_seq = "0011100101"
# sequence = test_seq

# define phi(x)
phi = lambda x: np.max(np.abs(x))


# cal the bound
def cal_bound(*args):
    assert len(args) == 2, "the length of input must eq 2"
    ar1 = args[0]
    ar2 = args[1]
    temp_bound1 = (ar1[0] - ar1[1]) / (ar2[0] - ar2[1])
    temp_bound2 = -(ar1[0] + ar1[1]) / (ar2[0] + ar2[1])
    return np.sort([temp_bound1, temp_bound2])


# define minimize(list)
def find_d(*args):
    """
    d: d is among the odd integers immediately
    less than(<) or greater than(>)
    (f1-f2)/(g1-g2) and -(f1,f2)/(g1+g2)
    """
    assert len(args) == 4, "It need tow args(f, g) to find d"
    lowwer_bound = args[2]+1 if args[2] % 2 == 0 else args[2]
    upper_bound = args[3]
    if lowwer_bound == upper_bound:
        return lowwer_bound
    d_range = np.arange(lowwer_bound, upper_bound, 2)[:, np.newaxis]
    # print(np.abs(args[0] + d*args[1]))
    tmp_phi = np.max(np.abs(args[0] + d_range*args[1]), 1)
    d = d_range[np.where(tmp_phi == np.min(tmp_phi))]
    return d[0]


def test(g, sequence):
    pad = 0
    for index, val in enumerate(sequence):
        pad += int(val) * 2**index

    if len(sequence) < 200:
        print("sequence: " + str(sequence))

    gt = pow(g[1]*pad, 1, (2**len(sequence))) % (2**len(sequence))
    rst = g[0] % (2**len(sequence))
    if rst != gt:
        print("Error with g: ", g)
    else:
        print("emmmm..Your result is correct!")


# find the first nonzero k(a_i)
for k, val in enumerate(sequence):
    if val == '1':
        break

# define alpha, f and g
a_k = 1
alpha = 2**k
f = np.array([0, 2])
g = np.array([2**k, 1])

i = k+1
while i < len(sequence):
    # alpha += a_k * (2**k)
    alpha += int(sequence[i]) * 2**i

    if pow(alpha*g[1] - g[0], 1, 2**(i+1)) % 2**(i+1) == 0:
        f *= 2

    elif phi(g) < phi(f):
        bound = cal_bound(f, g)
        d = find_d(f, g, bound[0], bound[1])
        g, f = f+d*g, 2*g

    else:
        bound = cal_bound(g, f)
        d = find_d(g, f, bound[0], bound[1])
        g, f = g+d*f, f*2
    i += 1

print("result_g: {}".format(g))

test(g, sequence)
